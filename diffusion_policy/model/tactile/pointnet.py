import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union




def random_rotation_z(points: torch.Tensor, max_angle: float = np.pi) -> torch.Tensor:
    """
    Apply random rotation around Z-axis to point cloud.
    
    Args:
        points: Point cloud tensor with shape (B, N, 3) or (N, 3)
        max_angle: Maximum rotation angle in radians
    
    Returns:
        Rotated point cloud with same shape as input
    """
    if len(points.shape) == 2:
        # Single point cloud (N, 3)
        theta = torch.rand(1, device=points.device) * 2 * max_angle - max_angle
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rot_matrix = torch.tensor([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], device=points.device, dtype=points.dtype).squeeze(0)
        
        return torch.matmul(points, rot_matrix.T)
    else:
        # Batch of point clouds (B, N, 3)
        B = points.shape[0]
        theta = torch.rand(B, device=points.device) * 2 * max_angle - max_angle
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # Create rotation matrices for each batch
        zeros = torch.zeros_like(cos_theta)
        ones = torch.ones_like(cos_theta)
        
        rot_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta, zeros], dim=1),
            torch.stack([sin_theta, cos_theta, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)  # (B, 3, 3)
        
        return torch.bmm(points, rot_matrices.transpose(1, 2))


def random_jitter(points: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
    """
    Add random Gaussian noise to point cloud.
    
    Args:
        points: Point cloud tensor with shape (B, N, 3) or (N, 3)
        sigma: Standard deviation of Gaussian noise
        clip: Maximum noise magnitude (clipping value)
    
    Returns:
        Jittered point cloud with same shape as input
    """
    noise = torch.randn_like(points) * sigma
    noise = torch.clamp(noise, -clip, clip)
    return points + noise


def random_scaling(points: torch.Tensor, scale_low: float = 0.8, scale_high: float = 1.25) -> torch.Tensor:
    """
    Apply random scaling to point cloud.
    
    Args:
        points: Point cloud tensor with shape (B, N, 3) or (N, 3)
        scale_low: Minimum scaling factor
        scale_high: Maximum scaling factor
    
    Returns:
        Scaled point cloud with same shape as input
    """
    if len(points.shape) == 2:
        # Single point cloud (N, 3)
        scale = torch.rand(1, device=points.device) * (scale_high - scale_low) + scale_low
        return points * scale
    else:
        # Batch of point clouds (B, N, 3)
        B = points.shape[0]
        scale = torch.rand(B, 1, 1, device=points.device) * (scale_high - scale_low) + scale_low
        return points * scale


def random_dropout(points: torch.Tensor, keep_ratio: float = 0.9) -> torch.Tensor:
    """
    Randomly dropout points from point cloud.
    
    Args:
        points: Point cloud tensor with shape (B, N, 3) or (N, 3)
        keep_ratio: Ratio of points to keep
    
    Returns:
        Point cloud with some points dropped, padded with zeros to maintain shape
    """
    if len(points.shape) == 2:
        # Single point cloud (N, 3)
        N = points.shape[0]
        keep_N = int(N * keep_ratio)
        if keep_N < N:
            indices = torch.randperm(N, device=points.device)[:keep_N]
            result = torch.zeros_like(points)
            result[:keep_N] = points[indices]
            return result
        return points
    else:
        # Batch of point clouds (B, N, 3)
        B, N, _ = points.shape
        keep_N = int(N * keep_ratio)
        if keep_N < N:
            result = torch.zeros_like(points)
            for b in range(B):
                indices = torch.randperm(N, device=points.device)[:keep_N]
                result[b, :keep_N] = points[b, indices]
            return result
        return points




class TNet(nn.Module):
    """T-Net for learning transformation matrices."""
    
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Remove dropout as per original paper
        # self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (B, C, N)
        Returns:
            Transformation matrix with shape (B, k, k)
        """
        B, _, N = x.shape
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Fully connected layers (no dropout as per original paper)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetEncoder(nn.Module):
    """PointNet encoder for extracting global features from point clouds."""
    
    def __init__(self, 
                 input_dim: int = 3,
                 output_dim: int = 1024,
                 use_tnet: bool = True,
                 use_feature_transform: bool = True,
                 use_batch_norm: bool = True):
        super(PointNetEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tnet = use_tnet
        self.use_feature_transform = use_feature_transform
        self.use_batch_norm = use_batch_norm
        
        # Input transform
        if self.use_tnet:
            self.input_transform = TNet(k=input_dim)
        
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
        else:
            self.bn1 = nn.LayerNorm(64)
            self.bn2 = nn.LayerNorm(128)  
            self.bn3 = nn.LayerNorm(1024)
        
        # Feature transform (only for 64-dim features after conv1)
        if self.use_feature_transform:
            self.feature_transform = TNet(k=64)
        
        # Final projection if needed
        if output_dim != 1024:
            self.final_proj = nn.Linear(1024, output_dim)
        else:
            self.final_proj = None
            
    def forward(self, x):
        """
        Args:
            x: Point cloud tensor with shape (B, N, input_dim)
        
        Returns:
            Global feature tensor with shape (B, output_dim)
        """
        # Ensure input is in (B, N, input_dim) format, then convert to (B, input_dim, N)
        if x.shape[-1] == self.input_dim:  # (B, N, C)
            x = x.transpose(1, 2)          # -> (B, C, N)
        
        B, C, N = x.shape
        assert C == self.input_dim, f"Expected input_dim={self.input_dim}, got {C}"
        
        # Input transformation (following standard paper implementation)
        if self.use_tnet:
            trans = self.input_transform(x)        # (B, C, C)
            x = x.transpose(2, 1)                  # (B, N, C)
            x = torch.bmm(x, trans)                # (B, N, C)
            x = x.transpose(2, 1)                  # (B, C, N)
        
        # First point-wise MLP
        x = self.conv1(x)  # (B, 64, N)
        if self.use_batch_norm:
            x = F.relu(self.bn1(x))
        else:
            x = x.transpose(1, 2)  # (B, N, 64) for LayerNorm
            x = F.relu(self.bn1(x))
            x = x.transpose(1, 2)  # (B, 64, N)
        
        # Feature transformation (only applied after first conv1)
        if self.use_feature_transform:
            trans_feat = self.feature_transform(x)  # (B, 64, 64)
            x = x.transpose(2, 1)                   # (B, N, 64)
            x = torch.bmm(x, trans_feat)            # (B, N, 64)
            x = x.transpose(2, 1)                   # (B, 64, N)
        
        # Remaining point-wise MLPs
        x = self.conv2(x)  # (B, 128, N)
        if self.use_batch_norm:
            x = F.relu(self.bn2(x))
        else:
            x = x.transpose(1, 2)  # (B, N, 128)
            x = F.relu(self.bn2(x))
            x = x.transpose(1, 2)  # (B, 128, N)
            
        x = self.conv3(x)  # (B, 1024, N)
        if self.use_batch_norm:
            x = self.bn3(x)
        else:
            x = x.transpose(1, 2)  # (B, N, 1024)
            x = self.bn3(x)
            x = x.transpose(1, 2)  # (B, 1024, N)
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=False)[0]  # (B, 1024)
        
        # Final projection if needed
        if self.final_proj is not None:
            x = self.final_proj(x)
        
        return x


class PointNetFeatureExtractor(nn.Module):
    """PointNet-based feature extractor for tactile point clouds."""
    
    def __init__(self,
                 num_points: int = 128,
                 input_dim: int = 3,
                 feature_dim: int = 512,
                 use_tnet: bool = True,
                 use_feature_transform: bool = True,
                 use_batch_norm: bool = True,  # True for BatchNorm, False for LayerNorm
                 # Data augmentation parameters
                 use_augmentation: bool = True,
                 aug_rotation: bool = True,
                 aug_jitter: bool = True,
                 aug_scaling: bool = True,
                 aug_dropout: bool = True,
                 rotation_angle: float = np.pi / 6,  # 30 degrees
                 jitter_sigma: float = 0.01,
                 jitter_clip: float = 0.02,
                 scale_low: float = 0.9,
                 scale_high: float = 1.1,
                 keep_ratio: float = 0.95):
        super(PointNetFeatureExtractor, self).__init__()
        
        self.num_points = num_points
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # Data augmentation settings
        self.use_augmentation = use_augmentation
        self.aug_rotation = aug_rotation
        self.aug_jitter = aug_jitter
        self.aug_scaling = aug_scaling
        self.aug_dropout = aug_dropout
        self.rotation_angle = rotation_angle
        self.jitter_sigma = jitter_sigma
        self.jitter_clip = jitter_clip
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.keep_ratio = keep_ratio
        
        # PointNet encoder
        self.pointnet = PointNetEncoder(
            input_dim=input_dim,
            output_dim=feature_dim,
            use_tnet=use_tnet,
            use_feature_transform=use_feature_transform,
            use_batch_norm=use_batch_norm
        )
        
    def forward(self, points):
        """
        Args:
            points: Point cloud tensor with shape (B, N, 3)
        
        Returns:
            Feature tensor with shape (B, feature_dim)
        """
        # Points are already sampled to consistent size in the dataset
        
        # Apply data augmentation during training
        if self.training and self.use_augmentation:
            # Apply individual augmentations based on settings
            if self.aug_rotation:
                points = random_rotation_z(points, self.rotation_angle)
            
            if self.aug_jitter:
                points = random_jitter(points, self.jitter_sigma, self.jitter_clip)
            
            if self.aug_scaling:
                points = random_scaling(points, self.scale_low, self.scale_high)
            
            if self.aug_dropout:
                points = random_dropout(points, self.keep_ratio)
        
        # Extract features using PointNet
        # Points should be (B, N, 3), PointNet expects this format
        features = self.pointnet(points)
        
        return features


class BimanualPointNetFeatureExtractor(nn.Module):
    """Bimanual PointNet feature extractor for dual-hand tactile sensing."""
    
    def __init__(self,
                 num_points: int = 128,
                 input_dim: int = 3,
                 feature_dim: int = 512,
                 use_tnet: bool = True,
                 use_feature_transform: bool = True,
                 use_batch_norm: bool = True,
                 fusion_method: str = 'concat',
                 # Data augmentation parameters
                 use_augmentation: bool = True,
                 aug_rotation: bool = True,
                 aug_jitter: bool = True,
                 aug_scaling: bool = True,
                 aug_dropout: bool = True,
                 rotation_angle: float = np.pi / 6,
                 jitter_sigma: float = 0.01,
                 jitter_clip: float = 0.02,
                 scale_low: float = 0.9,
                 scale_high: float = 1.1,
                 keep_ratio: float = 0.95):
        super(BimanualPointNetFeatureExtractor, self).__init__()
        
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        
        # Separate PointNet encoders for each hand
        self.left_pointnet = PointNetFeatureExtractor(
            num_points=num_points,
            input_dim=input_dim,
            feature_dim=feature_dim,
            use_tnet=use_tnet,
            use_feature_transform=use_feature_transform,
            use_batch_norm=use_batch_norm,
            use_augmentation=use_augmentation,
            aug_rotation=aug_rotation,
            aug_jitter=aug_jitter,
            aug_scaling=aug_scaling,
            aug_dropout=aug_dropout,
            rotation_angle=rotation_angle,
            jitter_sigma=jitter_sigma,
            jitter_clip=jitter_clip,
            scale_low=scale_low,
            scale_high=scale_high,
            keep_ratio=keep_ratio
        )
        
        self.right_pointnet = PointNetFeatureExtractor(
            num_points=num_points,
            input_dim=input_dim,
            feature_dim=feature_dim,
            use_tnet=use_tnet,
            use_feature_transform=use_feature_transform,
            use_batch_norm=use_batch_norm,
            use_augmentation=use_augmentation,
            aug_rotation=aug_rotation,
            aug_jitter=aug_jitter,
            aug_scaling=aug_scaling,
            aug_dropout=aug_dropout,
            rotation_angle=rotation_angle,
            jitter_sigma=jitter_sigma,
            jitter_clip=jitter_clip,
            scale_low=scale_low,
            scale_high=scale_high,
            keep_ratio=keep_ratio
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            self.output_dim = feature_dim * 2
            self.fusion = None
        elif fusion_method == 'add':
            self.output_dim = feature_dim
            self.fusion = None
        elif fusion_method == 'mlp':
            self.output_dim = feature_dim
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, left_points, right_points):
        """
        Args:
            left_points: Point cloud data for left hand
            right_points: Point cloud data for right hand
        
        Returns:
            Fused feature tensor
        """
        left_features = self.left_pointnet(left_points)
        right_features = self.right_pointnet(right_points)
        
        if self.fusion_method == 'concat':
            return torch.cat([left_features, right_features], dim=-1)
        elif self.fusion_method == 'add':
            return left_features + right_features
        elif self.fusion_method == 'mlp':
            concat_features = torch.cat([left_features, right_features], dim=-1)
            return self.fusion(concat_features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")