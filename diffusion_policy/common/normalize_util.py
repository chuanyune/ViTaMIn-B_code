from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np
import torch


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    # Formula: y = (x + offset) * scale
    # We want: x=input_min -> y=output_min, x=input_max -> y=output_max
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    
    # New formula: y = (x + offset) * scale
    # offset = -(input_min + input_max) / 2
    # scale = (output_max - output_min) / input_range
    offset = -(input_min + input_max) / 2
    scale = (output_max - output_min) / input_range
    offset[ignore_dim] = -input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    # Formula: y = (x + offset) * scale
    # For x in [0, 1] -> y in [-1, 1]
    # offset = -(0 + 1) / 2 = -0.5
    # scale = 2 / (1 - 0) = 2
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-0.5], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_identity_normalizer():
    # Input 0-1 then normalize
    scale = np.array([1], dtype=np.float32)
    offset = np.array([0], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_pointcloud_normalizer_from_stat(stat, 
                                       norm_method="unit_sphere", 
                                       center_method="centroid",
                                       scale_method="max_dist"):
    """
    Create a point cloud normalizer based on statistics.
    
    Args:
        stat: Statistics dictionary with keys 'min', 'max', 'mean', 'std'
        norm_method: "unit_sphere", "zero_mean", or "minmax"
        center_method: "centroid", "bbox_center", or "origin"
        scale_method: "max_dist", "std", or "bbox_diagonal"
    """
    if norm_method == "unit_sphere":
        # Center to centroid and scale to unit sphere
        if center_method == "centroid":
            offset = -stat['mean']  # Center at mean
        elif center_method == "bbox_center":
            offset = -(stat['min'] + stat['max']) / 2  # Center at bbox center
        else:  # origin
            offset = np.zeros_like(stat['mean'])
            
        if scale_method == "max_dist":
            # Scale by maximum distance from center
            centered_max = np.maximum(
                np.abs(stat['max'] + offset),
                np.abs(stat['min'] + offset)
            )
            scale_val = 1.0 / np.maximum(np.max(centered_max), 1e-8)
            scale = np.full_like(offset, scale_val)  # Broadcast scalar to match offset shape
        elif scale_method == "std":
            scale_val = 1.0 / np.maximum(np.max(stat['std']), 1e-8)
            scale = np.full_like(offset, scale_val)  # Broadcast scalar to match offset shape
        else:  # bbox_diagonal
            bbox_diag = np.linalg.norm(stat['max'] - stat['min'])
            scale_val = 1.0 / np.maximum(bbox_diag, 1e-8)
            scale = np.full_like(offset, scale_val)  # Broadcast scalar to match offset shape
            
    elif norm_method == "zero_mean":
        # Zero mean, unit variance
        offset = -stat['mean']
        scale = 1.0 / np.maximum(stat['std'], 1e-8)
        
    elif norm_method == "minmax":
        # Scale to [-1, 1] range
        offset = -(stat['min'] + stat['max']) / 2
        scale = 2.0 / np.maximum(stat['max'] - stat['min'], 1e-8)
        
    else:
        raise ValueError(f"Unknown norm_method: {norm_method}")
    
    # Ensure scale and offset have correct shape for point clouds
    # Convert to numpy arrays if not already
    if not isinstance(scale, np.ndarray):
        scale = np.array(scale)
    if not isinstance(offset, np.ndarray):
        offset = np.array(offset)
    
    # Ensure both are 1D arrays of shape (3,) for xyz coordinates
    if scale.ndim == 0:  # scalar
        scale = np.full(3, scale)
    if offset.ndim == 0:  # scalar
        offset = np.full(3, offset)
    
    # Debug: print shapes before creating normalizer
    if norm_method != "identity":
        print(f"[Point Cloud Normalizer] Creating normalizer:")
        print(f"  - scale shape: {scale.shape}, values: {scale}")
        print(f"  - offset shape: {offset.shape}, values: {offset}")
    
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat

def concatenate_normalizer(normalizers: list):
    scale = torch.concatenate([normalizer.params_dict['scale'] for normalizer in normalizers], axis=-1)
    offset = torch.concatenate([normalizer.params_dict['offset'] for normalizer in normalizers], axis=-1)
    input_stats_dict = dict_apply_reduce(
        [normalizer.params_dict['input_stats'] for normalizer in normalizers], 
        lambda x: torch.concatenate(x,axis=-1))
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=input_stats_dict
    )