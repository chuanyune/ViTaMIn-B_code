import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.tactile.mae import MAE_ViT, MAE_Encoder

logger = logging.getLogger(__name__)

class MultiModalObsEncoder(ModuleAttrMixin):
    def __init__(self,
                 shape_meta: dict,
                 image_model_name: str,
                 tactile_model_name: str,
                 pretrained: bool,
                 frozen: bool,
                 global_pool: str,
                 transforms: list,
                 # replace BatchNorm with GroupNorm
                 use_group_norm: bool = False,
                 # use single rgb model for all rgb inputs
                 share_rgb_model: bool = False,
                 # renormalize rgb input with imagenet normalization
                 # assuming input in [0,1]
                 imagenet_norm: bool = False,
                 downsample_ratio: int = 32,
                 position_encording: str = 'learnable',
                 mae_args: dict = dict(),
                ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        low_dim_keys = list()
        rgb_keys = list()
        tactile_keys = list()
        key_shape_map = dict()

        assert global_pool == ''

        # if image_model_name.startswith('vit'):
        if 'vit' in image_model_name.lower():
            
            self.image_model = timm.create_model(
                model_name=image_model_name,
                pretrained=pretrained,
                global_pool=global_pool,  # '' means no pooling
                num_classes=0  # remove classification layer
            ) # use pretrained image vit
        else:
            self.image_model = MAE_ViT(**mae_args).encoder.to(self.device)
            # if pretrained:
            #     model_dict = torch.load(image_model_name).encoder.state_dict()
            #     print('loading pretrained models from {}'.format(image_model_name))
            #     self.image_model.load_state_dict(model_dict, strict=True)

        self.tactile_model = MAE_ViT(**mae_args).encoder.to(self.device)
        # if pretrained:
        #     print('loading pretrained models from {}'.format(tactile_model_name))
        #     model_dict = torch.load(tactile_model_name).encoder.state_dict()
        #     self.tactile_model.load_state_dict(model_dict, strict=True)

        if frozen:
            assert pretrained
            for param in self.image_model.parameters():
                param.requires_grad = False
            for param in self.tactile_model.parameters():
                param.requires_grad = False

        if use_group_norm and not pretrained:
            self.image_model = replace_submodules(
                root_module=self.image_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )
            self.tactile_model = replace_submodules(
                root_module=self.tactile_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                             torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                             torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                         ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        self.image_transform = transform


        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                if 'tactile' in key:
                    tactile_keys.append(key)
                else:
                    rgb_keys.append(key)
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        tactile_keys = sorted(tactile_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('tactile keys:     ', tactile_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.image_model_name = image_model_name
        self.tactile_model_name = tactile_model_name
        self.shape_meta = shape_meta
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        return feature[:, 0, :]

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process image input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.image_transform(img)
            raw_feature = self.image_model(img)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # Find available tactile cameras dynamically
        tactile_keys = [k for k in obs_dict.keys() if 'tactile' in k]
        
        if tactile_keys:
            # Use first available tactile camera to get shape
            B, T = obs_dict[tactile_keys[0]].shape[:2]
            
            def resize_img(img):
                img = img.reshape(B * T, *img.shape[2:])
                resize = torchvision.transforms.Resize((112, 224), antialias=True)
                img = resize(img)
                return img

            # Try to find left and right tactile, or use available ones
            left_tactile_keys = [k for k in tactile_keys if 'left' in k]
            right_tactile_keys = [k for k in tactile_keys if 'right' in k]
            
            if left_tactile_keys and right_tactile_keys:
                # Both left and right available
                left_tactile = resize_img(obs_dict[left_tactile_keys[0]])
                right_tactile = resize_img(obs_dict[right_tactile_keys[0]])
                tactile = torch.cat([left_tactile, right_tactile], dim=2) # B*T, C, H, W
            elif len(tactile_keys) >= 2:
                # Use first two available tactile cameras
                tactile1 = resize_img(obs_dict[tactile_keys[0]])
                tactile2 = resize_img(obs_dict[tactile_keys[1]])
                tactile = torch.cat([tactile1, tactile2], dim=2) # B*T, C, H, W
            else:
                # Only one tactile camera, duplicate it
                single_tactile = resize_img(obs_dict[tactile_keys[0]])
                tactile = torch.cat([single_tactile, single_tactile], dim=2) # B*T, C, H, W
            
            tactile_feature = self.tactile_model.forward_representation(tactile)
            feature = self.aggregate_feature(tactile_feature)
            features.append(feature.reshape(B, -1))
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1

        return example_output.shape