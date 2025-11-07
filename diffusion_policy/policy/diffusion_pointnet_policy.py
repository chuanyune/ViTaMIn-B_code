from typing import Dict
import torch
import torch.nn.functional as F
import numpy as np
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.pointnet_multimodal_obs_encoder import PointNetMultiModalObsEncoder


class DiffusionPointNetPolicy(BaseImagePolicy):
    """
    Diffusion policy with PointNet multimodal observation encoder.
    
    Supports RGB images (ViT) + tactile images (ResNet) + point clouds (PointNet) + low-dimensional observations.
    """
    
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: PointNetMultiModalObsEncoder,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 input_pertub=0.1,
                 inpaint_fixed_action_prefix=False,
                 train_diffusion_n_samples=1,
                 # parameters passed to step
                 **kwargs
                 ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        # get feature dim from PointNet multimodal encoder
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self,
                           condition_data,
                           condition_mask,
                           local_cond=None,
                           global_cond=None,
                           generator=None,
                           # keyword arguments to scheduler.step
                           **kwargs
                           ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t,
                                 local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor = None) -> Dict[
        str, torch.Tensor]:
        """
        Predict action using multimodal observations (RGB + tactile images + point clouds + low-dim).
        
        Args:
            obs_dict: Dictionary containing:
                - RGB images (type: rgb) → processed by ViT
                - Tactile images (type: tactile_img) → processed by ResNet  
                - Point clouds (type: pc) → processed by PointNet
                - Low-dim data (type: low_dim) → used directly
            fixed_action_prefix: unnormalized action prefix
            
        Returns:
            Dictionary with "action" key containing predicted actions
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # extract multimodal features using PointNet encoder
        global_cond, repr_loss = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute training loss using multimodal observations.
        """
        # normalize input
        assert 'valid_mask' not in batch
        
        # Debug: Check batch data before normalization (only first iteration)
        if not hasattr(self, '_debug_batch_printed'):
            self._debug_batch_printed = True
            print(f"\n{'='*80}")
            print(f"[Training Batch - BEFORE Normalization]")
            print(f"Batch observation keys: {list(batch['obs'].keys())}")
            print(f"\n--- Point Cloud Data ---")
            for key, value in batch['obs'].items():
                if 'tactile_points' in key:
                    print(f"{key}:")
                    print(f"  shape: {value.shape}, dtype: {value.dtype}")
                    print(f"  range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
                    non_zero = (value != 0).any(dim=-1).sum().item()
                    total = value.shape[0] * value.shape[1] * value.shape[2]
                    print(f"  non-zero points: {non_zero}/{total}")
            
            print(f"\n--- Low-dim Data ---")
            for key, value in batch['obs'].items():
                if 'eef_pos' in key or 'gripper' in key:
                    print(f"{key}:")
                    print(f"  shape: {value.shape}, range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- Rotation Data ---")
            for key, value in batch['obs'].items():
                if 'rot_axis_angle' in key:
                    print(f"{key}:")
                    print(f"  shape: {value.shape}, range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- RGB Image Data ---")
            for key, value in batch['obs'].items():
                if 'rgb' in key:
                    print(f"{key}:")
                    print(f"  shape: {value.shape}, range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- Action Data ---")
            print(f"action:")
            print(f"  shape: {batch['action'].shape}, range: [{batch['action'].min().item():.4f}, {batch['action'].max().item():.4f}]")
            print(f"  mean: {batch['action'].mean().item():.4f}, std: {batch['action'].std().item():.4f}")
            print(f"{'='*80}\n")
        
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        # Debug: Check normalized data (only first iteration)
        if not hasattr(self, '_debug_normalized_printed'):
            self._debug_normalized_printed = True
            print(f"\n{'='*80}")
            print(f"[Training Batch - AFTER Normalization]")
            
            print(f"\n--- Point Cloud Data (Normalized) ---")
            for key, value in nobs.items():
                if 'tactile_points' in key:
                    print(f"{key}:")
                    print(f"  shape: {value.shape}, dtype: {value.dtype}")
                    print(f"  range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- Low-dim Data (Normalized) ---")
            for key, value in nobs.items():
                if 'eef_pos' in key or 'gripper' in key:
                    print(f"{key}:")
                    print(f"  range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- Rotation Data (Normalized) ---")
            for key, value in nobs.items():
                if 'rot_axis_angle' in key:
                    print(f"{key}:")
                    print(f"  range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- RGB Image Data (Normalized) ---")
            for key, value in nobs.items():
                if 'rgb' in key:
                    print(f"{key}:")
                    print(f"  range: [{value.min().item():.4f}, {value.max().item():.4f}]")
                    print(f"  mean: {value.mean().item():.4f}, std: {value.std().item():.4f}")
            
            print(f"\n--- Action Data (Normalized) ---")
            print(f"action:")
            print(f"  shape: {nactions.shape}")
            print(f"  range: [{nactions.min().item():.4f}, {nactions.max().item():.4f}]")
            print(f"  mean: {nactions.mean().item():.4f}, std: {nactions.std().item():.4f}")
            print(f"{'='*80}\n")

        assert self.obs_as_global_cond
        # extract multimodal features
        global_cond, repr_loss = self.obs_encoder(nobs)
        
        # Debug: Check extracted features (only first iteration)
        if not hasattr(self, '_debug_features_printed'):
            self._debug_features_printed = True
            print(f"\n{'='*80}")
            print(f"[Feature Extraction] Extracted global_cond shape: {global_cond.shape}")
            print(f"[Feature Extraction] global_cond range: [{global_cond.min().item():.4f}, {global_cond.max().item():.4f}]")
            print(f"[Feature Extraction] global_cond mean: {global_cond.mean().item():.4f}, std: {global_cond.std().item():.4f}")
            print(f"[Feature Extraction] repr_loss: {repr_loss}")
            print(f"{'='*80}\n")

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(global_cond,
                                                  repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions,
                                               repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the actions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additional noise to alleviate exposure bias
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return {'loss': loss + repr_loss, 'bc_loss': loss, 'repr_loss': repr_loss}

    def forward(self, batch):
        return self.compute_loss(batch)
