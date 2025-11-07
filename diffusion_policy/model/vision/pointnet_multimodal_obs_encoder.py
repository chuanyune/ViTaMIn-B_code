import copy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import numpy as np

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.tactile.pointnet import PointNetFeatureExtractor
from diffusion_policy.model.common.low_dim_encoder import create_low_dim_encoder

logger = logging.getLogger(__name__)


class MlpHead(nn.Module):
    """Multi-layer perceptron head for feature transformation."""
    def __init__(self, input_dim, hidden1_dim=256, hidden2_dim=128, output_dim=768):
        super(MlpHead, self).__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp_layers(x)


def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class PointNetMultiModalObsEncoder(ModuleAttrMixin):
    """
    Unified MultiModal observation encoder supporting three configurations:
    1. Pure Vision: RGB images (ViT) + low-dimensional observations
    2. Vision + Tactile Images: RGB images (ViT) + tactile images (ResNet) + low-dimensional observations  
    3. Vision + Tactile Point Clouds: RGB images (ViT) + tactile point clouds (PointNet) + low-dimensional observations
    
    Features:
    - Flexible modality support based on observation types in shape_meta
    - Configurable sharing of tactile encoders (shared vs separate)
    - Proper feature aggregation and dimension handling
    - Transform support for data augmentation
    """
    
    def __init__(self,
                 shape_meta: dict,
                 # Vision model parameters
                 image_model_name: str = 'vit_base_patch16_224.dino',
                 pretrained: bool = True,
                 frozen: bool = False,
                 global_pool: str = '',
                 # Transform parameters
                 transforms: list = None,
                 imagenet_norm: bool = True,
                 # Tactile image parameters
                 tactile_model_name: str = 'resnet18.a1_in1k',
                 share_tactile_model: bool = True,
                 use_resnet_for_tactile: bool = True,
                 # PointNet parameters
                 pointnet_num_points: int = 128,
                 pointnet_feature_dim: int = 768,
                 pointnet_use_batch_norm: bool = True,
                 share_pointnet_model: bool = True,
                 # Feature aggregation parameters
                 feature_aggregation: str = 'attention_pool_2d',
                 downsample_ratio: int = 32,
                 # Other parameters
                 use_group_norm: bool = True,
                 image_size: int = 224,
                 feature_dim: int = 768,
                 # MlpHead parameters for tactile ResNet
                 tactile_mlp_hidden1_dim: int = 256,
                 tactile_mlp_hidden2_dim: int = 128,
                 tactile_mlp_output_dim: int = 768,
                 # PointNet augmentation parameters
                 pointnet_use_augmentation: bool = True,
                 pointnet_aug_rotation: bool = True,
                 pointnet_aug_jitter: bool = True,
                 pointnet_aug_scaling: bool = False,
                 pointnet_aug_dropout: bool = False,
                 pointnet_rotation_angle: float = 0.5236,  # π/6 = 30 degrees
                 pointnet_jitter_sigma: float = 0.005,
                 pointnet_jitter_clip: float = 0.01,
                 # PointNet normalization parameters
                 pointnet_use_normalization: bool = True,
                 pointnet_norm_method: str = "unit_sphere",
                 pointnet_norm_per_sensor: bool = False,
                 pointnet_norm_center_method: str = "centroid",
                 pointnet_norm_scale_method: str = "max_dist",
                 # Low-dimensional encoder parameters
                 use_low_dim_encoder: bool = False,
                 low_dim_hidden_dims: list = None,
                 low_dim_output_dim: int = 256,
                 low_dim_dropout_rate: float = 0.1,
                 low_dim_use_batch_norm: bool = True,
                 low_dim_activation: str = 'relu',
               ):
        super().__init__()
        
        # Store configuration
        self.image_model_name = image_model_name
        self.tactile_model_name = tactile_model_name
        self.share_tactile_model = share_tactile_model
        self.use_resnet_for_tactile = use_resnet_for_tactile
        self.pointnet_num_points = pointnet_num_points
        self.pointnet_feature_dim = pointnet_feature_dim
        self.pointnet_use_batch_norm = pointnet_use_batch_norm
        self.share_pointnet_model = share_pointnet_model
        self.feature_aggregation = feature_aggregation
        self.downsample_ratio = downsample_ratio
        self.feature_dim = feature_dim
        self.shape_meta = shape_meta
        # MlpHead configuration
        self.tactile_mlp_hidden1_dim = tactile_mlp_hidden1_dim
        self.tactile_mlp_hidden2_dim = tactile_mlp_hidden2_dim
        self.tactile_mlp_output_dim = tactile_mlp_output_dim
        
        # Low-dimensional encoder configuration
        self.use_low_dim_encoder = use_low_dim_encoder
        self.low_dim_hidden_dims = low_dim_hidden_dims or [256, 512]
        self.low_dim_output_dim = low_dim_output_dim
        self.low_dim_dropout_rate = low_dim_dropout_rate
        self.low_dim_use_batch_norm = low_dim_use_batch_norm
        self.low_dim_activation = low_dim_activation
        
        # Initialize key collections
        low_dim_keys = list()
        rgb_keys = list()
        tactile_img_keys = list()
        tactile_pc_keys = list()
        key_shape_map = dict()

        assert global_pool == ''
        
        # Parse observation keys by type
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'tactile_img':
                tactile_img_keys.append(key)
            elif obs_type == 'pc':
                tactile_pc_keys.append(key)
            elif obs_type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        # Sort keys for consistent ordering
        rgb_keys = sorted(rgb_keys)
        tactile_img_keys = sorted(tactile_img_keys)
        tactile_pc_keys = sorted(tactile_pc_keys)
        low_dim_keys = sorted(low_dim_keys)
        
        print(f'RGB keys: {rgb_keys}')
        print(f'Tactile image keys: {tactile_img_keys}')
        print(f'Tactile point cloud keys: {tactile_pc_keys}')
        print(f'Low-dim keys: {low_dim_keys}')

        # Store keys for later use
        self.rgb_keys = rgb_keys
        self.tactile_img_keys = tactile_img_keys
        self.tactile_pc_keys = tactile_pc_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        # Create RGB vision encoders
        self.rgb_encoders = nn.ModuleDict()
        if rgb_keys:
            for rgb_key in rgb_keys:
                self.rgb_encoders[rgb_key] = timm.create_model(
                    model_name=image_model_name,
                    pretrained=pretrained,
                    global_pool=global_pool,
                    num_classes=0
                )
                print(f"Created ViT encoder for {rgb_key}")

        # Create tactile image encoders
        self.tactile_img_encoders = nn.ModuleDict()
        self.tactile_mlp_heads = nn.ModuleDict()  # Store MLP heads for tactile encoders
        if tactile_img_keys:
            if share_tactile_model:
                # Create shared tactile model
                model_name = tactile_model_name if use_resnet_for_tactile else image_model_name
                self.shared_tactile_model = timm.create_model(
                    model_name=model_name,
                    pretrained=pretrained,
                    global_pool=global_pool,
                    num_classes=0
                )
                print(f"Created shared {'ResNet' if use_resnet_for_tactile else 'ViT'} encoder for tactile images")
                
                # Add MlpHead for shared tactile model if using ResNet
                if use_resnet_for_tactile:
                    # Get the feature dimension of the backbone
                    if 'resnet18' in tactile_model_name:
                        backbone_dim = 512
                    elif 'resnet34' in tactile_model_name:
                        backbone_dim = 512
                    elif 'resnet50' in tactile_model_name:
                        backbone_dim = 2048
                    else:
                        backbone_dim = 512  # default
                    
                    self.shared_tactile_mlp_head = MlpHead(
                        input_dim=backbone_dim,
                        hidden1_dim=tactile_mlp_hidden1_dim,
                        hidden2_dim=tactile_mlp_hidden2_dim,
                        output_dim=tactile_mlp_output_dim
                    )
                    print(f"Created shared MlpHead for tactile: {backbone_dim} -> {tactile_mlp_output_dim}")
            else:
                # Create separate tactile encoders
                for tactile_key in tactile_img_keys:
                    model_name = tactile_model_name if use_resnet_for_tactile else image_model_name
                    self.tactile_img_encoders[tactile_key] = timm.create_model(
                        model_name=model_name,
                        pretrained=pretrained,
                        global_pool=global_pool,
                        num_classes=0
                    )
                    print(f"Created {'ResNet' if use_resnet_for_tactile else 'ViT'} encoder for {tactile_key}")
                    
                    # Add MlpHead for each tactile encoder if using ResNet
                    if use_resnet_for_tactile:
                        # Get the feature dimension of the backbone
                        if 'resnet18' in tactile_model_name:
                            backbone_dim = 512
                        elif 'resnet34' in tactile_model_name:
                            backbone_dim = 512
                        elif 'resnet50' in tactile_model_name:
                            backbone_dim = 2048
                        else:
                            backbone_dim = 512  # default
                        
                        self.tactile_mlp_heads[tactile_key] = MlpHead(
                            input_dim=backbone_dim,
                            hidden1_dim=tactile_mlp_hidden1_dim,
                            hidden2_dim=tactile_mlp_hidden2_dim,
                            output_dim=tactile_mlp_output_dim
                        )
                        print(f"Created MlpHead for {tactile_key}: {backbone_dim} -> {tactile_mlp_output_dim}")

        # Create PointNet encoders
        self.pointnet_encoders = nn.ModuleDict()
        if tactile_pc_keys:
            if share_pointnet_model:
                # Create shared PointNet model
                self.shared_pointnet_model = PointNetFeatureExtractor(
                    num_points=pointnet_num_points,
                    feature_dim=pointnet_feature_dim,
                    use_batch_norm=pointnet_use_batch_norm,
                    # Pass augmentation parameters
                    use_augmentation=pointnet_use_augmentation,
                    aug_rotation=pointnet_aug_rotation,
                    aug_jitter=pointnet_aug_jitter,
                    aug_scaling=pointnet_aug_scaling,
                    aug_dropout=pointnet_aug_dropout,
                    rotation_angle=pointnet_rotation_angle,
                    jitter_sigma=pointnet_jitter_sigma,
                    jitter_clip=pointnet_jitter_clip
                )
                print(f"Created shared PointNet encoder for tactile point clouds")
                print(f"  Augmentation: rotation={pointnet_aug_rotation}, jitter={pointnet_aug_jitter}, scaling={pointnet_aug_scaling}, dropout={pointnet_aug_dropout}")
            else:
                # Create separate PointNet encoders
                for pc_key in tactile_pc_keys:
                    self.pointnet_encoders[pc_key] = PointNetFeatureExtractor(
                        num_points=pointnet_num_points,
                        feature_dim=pointnet_feature_dim,
                        use_batch_norm=pointnet_use_batch_norm,
                        # Pass augmentation parameters
                        use_augmentation=pointnet_use_augmentation,
                        aug_rotation=pointnet_aug_rotation,
                        aug_jitter=pointnet_aug_jitter,
                        aug_scaling=pointnet_aug_scaling,
                        aug_dropout=pointnet_aug_dropout,
                        rotation_angle=pointnet_rotation_angle,
                        jitter_sigma=pointnet_jitter_sigma,
                        jitter_clip=pointnet_jitter_clip
                    )
                    print(f"Created PointNet encoder for {pc_key}")
                print(f"  Augmentation: rotation={pointnet_aug_rotation}, jitter={pointnet_aug_jitter}, scaling={pointnet_aug_scaling}, dropout={pointnet_aug_dropout}")

        # Freeze parameters if specified
        if frozen:
            assert pretrained
            # Freeze RGB encoders
            for rgb_encoder in self.rgb_encoders.values():
                for param in rgb_encoder.parameters():
                    param.requires_grad = False
            # Freeze tactile image encoders
            if tactile_img_keys:
                if share_tactile_model:
                    for param in self.shared_tactile_model.parameters():
                        param.requires_grad = False
                else:
                    for tactile_encoder in self.tactile_img_encoders.values():
                        for param in tactile_encoder.parameters():
                            param.requires_grad = False
            # Freeze PointNet encoders
            if tactile_pc_keys:
                if share_pointnet_model:
                    for param in self.shared_pointnet_model.parameters():
                        param.requires_grad = False
                else:
                    for pointnet_encoder in self.pointnet_encoders.values():
                        for param in pointnet_encoder.parameters():
                            param.requires_grad = False

        # Apply GroupNorm if specified
        if use_group_norm and not pretrained:
            # Apply to RGB encoders
            for key, rgb_encoder in self.rgb_encoders.items():
                self.rgb_encoders[key] = replace_submodules(
                    root_module=rgb_encoder,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                        num_channels=x.num_features)
                )
            # Apply to tactile image encoders
            if tactile_img_keys:
                if share_tactile_model:
                    self.shared_tactile_model = replace_submodules(
                        root_module=self.shared_tactile_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                            num_channels=x.num_features)
                    )
                else:
                    for tactile_key in tactile_img_keys:
                        self.tactile_img_encoders[tactile_key] = replace_submodules(
                            root_module=self.tactile_img_encoders[tactile_key],
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                                num_channels=x.num_features)
                        )

        # Setup image transforms
        image_shape = None
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
                
        # Setup training transforms
        if transforms is not None and len(transforms) > 0 and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
            if imagenet_norm:
                transforms = transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        # Setup evaluation transforms
        eval_transforms = None
        if transforms is not None:
            eval_transforms = [torchvision.transforms.Resize(size=image_shape[0], antialias=True)]
            if imagenet_norm:
                eval_transforms = eval_transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        eval_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*eval_transforms)

        self.image_train_transform = train_transform
        self.image_eval_transform = eval_transform
        
        print('Train transforms:', train_transform)
        print('Eval transforms:', eval_transform)

        # Setup feature aggregation for attention pooling if needed
        if feature_aggregation == 'attention_pool_2d' and rgb_keys:
            # Calculate feature map shape for attention pooling
            if 'vit' in image_model_name.lower():
                # For ViT models, calculate patch grid size
                patch_size = 16  # Common patch size for ViT
                feature_map_size = image_size // patch_size  # e.g., 224 // 16 = 14
                embed_dim = feature_dim  # e.g., 768
                
                self.attention_pool_2d = AttentionPool2d(
                    spacial_dim=feature_map_size,
                    embed_dim=embed_dim,
                    num_heads=embed_dim // 64,
                    output_dim=embed_dim
                )
                print(f'Created attention pooling with spatial_dim={feature_map_size}, embed_dim={embed_dim}')
        
        # Initialize low-dimensional encoder if enabled
        self.low_dim_encoder = None
        if use_low_dim_encoder and low_dim_keys:
            # Calculate total low-dimensional input dimension
            low_dim_input_dim = 0
            for key in low_dim_keys:
                shape = key_shape_map[key]
                # Each observation has horizon steps, so multiply by horizon
                horizon = obs_shape_meta[key]['horizon']
                low_dim_input_dim += np.prod(shape) * horizon
            
            print(f'Low-dim input dimension: {low_dim_input_dim}')
            
            # Create low-dimensional encoder
            self.low_dim_encoder = create_low_dim_encoder(
                input_dim=low_dim_input_dim,
                hidden_dims=self.low_dim_hidden_dims,
                output_dim=low_dim_output_dim,
                dropout_rate=low_dim_dropout_rate,
                use_batch_norm=low_dim_use_batch_norm,
                activation=low_dim_activation
            )
            print(f'Created low-dimensional MLP encoder: {low_dim_input_dim} -> {low_dim_output_dim}')
        
        logger.info(
            "Number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature, model_type='vit', model=None, tactile_key=None):
        """Aggregate features from different types of models."""
        if model_type == 'resnet':
            # ResNet output: (B, channels, H, W) -> global average pool
            if len(feature.shape) == 4:  # (B, C, H, W)
                feature = F.adaptive_avg_pool2d(feature, (1, 1))  # -> (B, C, 1, 1)
                feature = feature.squeeze(-1).squeeze(-1)  # -> (B, C)
            elif len(feature.shape) == 2:  # Already (B, feature_dim)
                pass
            else:
                feature = feature.view(feature.shape[0], -1)
            
            # Apply MlpHead for tactile ResNet (always applied when using ResNet)
            if tactile_key is not None and self.use_resnet_for_tactile:
                if self.share_tactile_model and hasattr(self, 'shared_tactile_mlp_head'):
                    feature = self.shared_tactile_mlp_head(feature)
                elif tactile_key in self.tactile_mlp_heads:
                    feature = self.tactile_mlp_heads[tactile_key](feature)
            
            return feature
        elif model_type == 'vit':
            if 'clip' in getattr(self, 'image_model_name', '').lower():
                # CLIP models: use CLS token
                return feature[:, 0, :]
            else:
                # Standard ViT: use global pooling
                if model is not None:
                    num_prefix_tokens = getattr(model, 'num_prefix_tokens', 1)
                else:
                    num_prefix_tokens = 1
                feature = global_pool_nlc(feature, pool_type='token', num_prefix_tokens=num_prefix_tokens)
                return feature
        elif model_type == 'pointnet':
            # PointNet already outputs (B, feature_dim)
            return feature
        else:
            # Fallback: handle by shape
            if len(feature.shape) == 3:
                # (B, seq_len, feature_dim) -> take CLS token or pool
                return feature[:, 0, :]
            elif len(feature.shape) == 2:
                # Already 2D: (B, feature_dim) -> use as is
                return feature
            else:
                # Handle other cases by flattening
                return feature.view(feature.shape[0], -1)

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # Process RGB images
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T, C, H, W = img.shape
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])

            # Apply horizontal flip for right cameras (data augmentation)
            if 'right' in key:
                img = torch.flip(img, dims=[-1])
            
            # Apply transforms based on training/evaluation mode
            if self.training:
                img = self.image_train_transform(img)
            else:
                img = self.image_eval_transform(img)

            # Forward through RGB encoder
            raw_feature = self.rgb_encoders[key](img)
            feature = self.aggregate_feature(raw_feature, model_type='vit', model=self.rgb_encoders[key])
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # Process tactile images
        for key in self.tactile_img_keys:
            if key in obs_dict:
                img = obs_dict[key]
                B, T, C, H, W = img.shape
                assert B == batch_size
                assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B * T, *img.shape[2:])

                # Apply horizontal flip for right tactile sensors
                if 'right' in key:
                    img = torch.flip(img, dims=[-1])
                
                # Apply transforms
                if self.training:
                    img = self.image_train_transform(img)
                else:
                    img = self.image_eval_transform(img)

                # Forward through tactile encoder
                if self.share_tactile_model:
                    raw_feature = self.shared_tactile_model(img)
                else:
                    raw_feature = self.tactile_img_encoders[key](img)
                
                # Aggregate features based on model type
                model_type = 'resnet' if self.use_resnet_for_tactile else 'vit'
                if self.share_tactile_model:
                    model = self.shared_tactile_model
                else:
                    model = self.tactile_img_encoders[key]
                    
                feature = self.aggregate_feature(raw_feature, model_type=model_type, model=model, tactile_key=key)
                assert len(feature.shape) == 2 and feature.shape[0] == B * T
                features.append(feature.reshape(B, -1))
            else:
                raise ValueError(f"Expected tactile image key '{key}' not found in observation data. "
                                f"Available keys: {list(obs_dict.keys())}. "
                                f"This indicates a mismatch between the task configuration and actual data. "
                                f"Please ensure all required tactile sensors are properly configured and providing data.")

        # Process point clouds with PointNet
        if self.tactile_pc_keys:
            pc_features = self._process_pointcloud(obs_dict, batch_size)
            if pc_features is not None:
                features.append(pc_features)

        # Process low-dimensional observations
        if self.low_dim_keys:
            if self.low_dim_encoder is not None:
                # Use dedicated low-dimensional encoder
                low_dim_data_list = []
                for key in self.low_dim_keys:
                    data = obs_dict[key]
                    B, T = data.shape[:2]
                    assert B == batch_size
                    assert data.shape[2:] == self.key_shape_map[key]
                    low_dim_data_list.append(data.reshape(B, -1))
                
                # Concatenate all low-dimensional data
                concatenated_low_dim = torch.cat(low_dim_data_list, dim=-1)
                
                # Encode using the low-dimensional encoder
                low_dim_encoded = self.low_dim_encoder(concatenated_low_dim)
                features.append(low_dim_encoded)
            else:
                # Original simple concatenation approach
                for key in self.low_dim_keys:
                    data = obs_dict[key]
                    B, T = data.shape[:2]
                    assert B == batch_size
                    assert data.shape[2:] == self.key_shape_map[key]
                    low_dim_feature = data.reshape(B, -1)
                    features.append(low_dim_feature)

        # Concatenate all features
        if features:
            result = torch.cat(features, dim=-1)
        else:
            result = torch.zeros(batch_size, 1, device=self.device)


        # Return result and dummy repr_loss for compatibility
        repr_loss = torch.tensor(0.0, device=self.device)
        return result, repr_loss

    def _process_pointcloud(self, obs_dict, batch_size):
        """Process point clouds using PointNet encoders."""
        pc_keys_in_obs = [k for k in self.tactile_pc_keys if k in obs_dict]
        
        if not pc_keys_in_obs:
            return None

        all_features = []
        
        # Debug: Print once per forward pass
        if not hasattr(self, '_debug_pointcloud_printed'):
            self._debug_pointcloud_printed = True
            print(f"\n{'='*80}")
            print(f"[PointNet Encoder] Processing {len(pc_keys_in_obs)} point cloud sensors")
            print(f"[PointNet Encoder] Keys: {pc_keys_in_obs}")
            print(f"[PointNet Encoder] Shared model: {self.share_pointnet_model}")
            print(f"[PointNet Encoder] Expected num_points: {self.pointnet_num_points}")
        
        # Process each point cloud key
        for idx, key in enumerate(pc_keys_in_obs):
            points = obs_dict[key]
            
            # Handle tensor input (both training and inference use tensors now)
            if not isinstance(points, torch.Tensor):
                # Convert numpy array to tensor if needed
                points = torch.from_numpy(points.astype(np.float32))
            
            # Determine batch size and temporal dimension
            if len(points.shape) == 4:
                # Batch case: (B, T, num_points, 3)
                B, T, num_points, _ = points.shape
            elif len(points.shape) == 3:
                # Single batch case: (T, num_points, 3)
                T, num_points, _ = points.shape
                B = batch_size
                # Add batch dimension
                points = points.unsqueeze(0)  # (1, T, num_points, 3)
            else:
                raise ValueError(f"Unexpected point cloud shape: {points.shape}")
            
            assert B == batch_size
            
            # Reshape to (B*T, num_points, 3) for PointNet processing
            points_flat = points.reshape(B * T, num_points, 3)
            
            # Move to correct device
            points_flat = points_flat.to(device=self.device)
            
            # Debug: Print details for first sensor
            if not hasattr(self, '_debug_pointcloud_printed'):
                if idx == 0:
                    print(f"\n[PointNet Encoder] First sensor ({key}):")
                    print(f"  - Input shape: (B={B}, T={T}, N={num_points}, 3)")
                    print(f"  - Flattened shape: {points_flat.shape}")
                    print(f"  - Point range: [{points_flat.min().item():.4f}, {points_flat.max().item():.4f}]")
                    non_zero = (points_flat != 0).any(dim=-1).sum().item()
                    print(f"  - Non-zero points: {non_zero}/{points_flat.shape[0] * points_flat.shape[1]}")
            
            # Forward through PointNet
            if self.share_pointnet_model:
                raw_features = self.shared_pointnet_model(points_flat)  # (B*T, feature_dim)
            else:
                raw_features = self.pointnet_encoders[key](points_flat)  # (B*T, feature_dim)
            
            # Debug: Print features for first sensor
            if not hasattr(self, '_debug_pointcloud_printed'):
                if idx == 0:
                    print(f"  - Raw features shape: {raw_features.shape}")
                    print(f"  - Raw features range: [{raw_features.min().item():.4f}, {raw_features.max().item():.4f}]")
            
            # Aggregate features
            features_bt = self.aggregate_feature(raw_features, model_type='pointnet')
            
            # Reshape back to (B, T*feature_dim)
            key_features = features_bt.reshape(B, -1)
            all_features.append(key_features)
            
            # Debug: Print aggregated features for first sensor
            if not hasattr(self, '_debug_pointcloud_printed'):
                if idx == 0:
                    print(f"  - Aggregated features shape: {key_features.shape}")
                    print(f"  - Aggregated features range: [{key_features.min().item():.4f}, {key_features.max().item():.4f}]")
        
        if all_features:
            # Concatenate features from all point cloud keys
            result = torch.cat(all_features, dim=1)
            
            # Debug: Print final concatenated features
            if not hasattr(self, '_debug_pointcloud_printed'):
                print(f"\n[PointNet Encoder] Final concatenated features:")
                print(f"  - Shape: {result.shape}")
                print(f"  - Range: [{result.min().item():.4f}, {result.max().item():.4f}]")
                print(f"{'='*80}\n")
            
            return result
        
        return None

    def _points_to_tensor(self, points):
        """Convert point cloud to tensor."""
        if isinstance(points, np.ndarray) and len(points.shape) == 2 and points.shape[0] > 0:
            points_tensor = torch.from_numpy(points.astype(np.float32)).to(self.device)
            if points_tensor.shape[1] == 3:
                # Points are already sampled to correct size in dataset
                if points_tensor.shape[0] == self.pointnet_num_points:
                    return points_tensor
                elif points_tensor.shape[0] > self.pointnet_num_points:
                    # If too many points, take first N
                    return points_tensor[:self.pointnet_num_points]
                else:
                    # If too few points, pad with zeros
                    padded = torch.zeros(self.pointnet_num_points, 3, dtype=torch.float32, device=self.device)
                    padded[:points_tensor.shape[0]] = points_tensor
                    return padded
        
        # Fallback: create dummy points
        return torch.zeros(self.pointnet_num_points, 3, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def output_shape(self):
        """Compute output shape by running a forward pass with dummy data."""
        was_training = self.training
        self.eval()
        
        try:
            example_obs_dict = dict()
            obs_shape_meta = self.shape_meta['obs']
            
            for key, attr in obs_shape_meta.items():
                shape = tuple(attr['shape'])
                obs_type = attr.get('type', 'low_dim')
                
                if obs_type == 'pc':
                    # Create dummy point cloud data with fixed tensor shape
                    this_obs = torch.zeros(
                        (1, attr['horizon'], self.pointnet_num_points, 3),
                        dtype=torch.float32,
                        device=self.device
                    )
                elif obs_type == 'rgb' or obs_type == 'tactile_img':
                    # Create dummy image data (RGB or tactile images)
                    this_obs = torch.zeros(
                        (1, attr['horizon']) + shape,
                        dtype=self.dtype,
                        device=self.device
                    )
                else:
                    # Low-dimensional data
                    this_obs = torch.zeros(
                        (1, attr['horizon']) + shape,
                        dtype=self.dtype,
                        device=self.device
                    )
                
                example_obs_dict[key] = this_obs
            
            example_output, repr_loss = self.forward(example_obs_dict)
            assert len(example_output.shape) == 2
            assert example_output.shape[0] == 1

            return example_output.shape
        finally:
            if was_training:
                self.train()
