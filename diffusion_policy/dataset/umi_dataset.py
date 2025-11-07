import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat,
    get_pointcloud_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from utils.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()

class UmiDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        train_ratio: float=1.0,
        max_duration: Optional[float]=None,
        # Point cloud normalization parameters
        pointnet_use_normalization: bool=False,
        pointnet_norm_method: str="unit_sphere",
        pointnet_norm_per_sensor: bool=False,
        pointnet_norm_center_method: str="centroid",
        pointnet_norm_scale_method: str="max_dist"
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        # Point cloud normalization parameters
        self.pointnet_use_normalization = pointnet_use_normalization
        self.pointnet_norm_method = pointnet_norm_method
        self.pointnet_norm_per_sensor = pointnet_norm_per_sensor
        self.pointnet_norm_center_method = pointnet_norm_center_method
        self.pointnet_norm_scale_method = pointnet_norm_scale_method
        
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        tactile_img_keys = list()  # Tactile image keys
        tactile_points_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'tactile_img':  # Handle tactile image type
                tactile_img_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'pc':
                tactile_points_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        if train_ratio < 1.0:
            rng = np.random.RandomState(seed)
            train_indices = np.where(train_mask)[0]
            n_train = int(len(train_indices) * train_ratio)
            selected_indices = rng.choice(train_indices, 
                                        size=n_train, 
                                        replace=False)
            new_train_mask = np.zeros_like(train_mask)
            new_train_mask[selected_indices] = True
            train_mask = new_train_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_down_sample_steps=key_down_sample_steps,
            tactile_img_keys=tactile_img_keys,  # Tactile image keys
            tactile_points_keys=tactile_points_keys,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.tactile_img_keys = tactile_img_keys  # Save tactile image keys
        self.tactile_points_keys = tactile_points_keys
        self.key_horizon = key_horizon
        
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_down_sample_steps=self.key_down_sample_steps,
            tactile_img_keys=self.tactile_img_keys,  # Tactile image keys
            tactile_points_keys=self.tactile_points_keys,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:  # Where is this used?
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + self.tactile_points_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key].numpy()))
            for key in self.tactile_points_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key].numpy()))
            data_cache['action'].append(copy.deepcopy(batch['action'].numpy()))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key], axis=0)
            if key not in self.tactile_points_keys:  # Skip shape assertions for tactile points
                assert data_cache[key].shape[0] == len(self.sampler)
                assert len(data_cache[key].shape) == 3
                B, T, D = data_cache[key].shape
                if not self.temporally_independent_normalization:
                    data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        print(f"\n{'='*80}")
        print(f"[Normalizer Creation] Creating action normalizer...")
        print(f"Action shape: {data_cache['action'].shape}")
        print(f"Action range: [{data_cache['action'].min():.4f}, {data_cache['action'].max():.4f}]")
        print(f"Action mean: {data_cache['action'].mean():.4f}, std: {data_cache['action'].std():.4f}")
        
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            pos_stat = array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])
            print(f"Robot{i} action position - min: {pos_stat['min']}, max: {pos_stat['max']}")
            action_normalizers.append(get_range_normalizer_from_stat(pos_stat))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            gripper_stat = array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])
            print(f"Robot{i} action gripper - min: {gripper_stat['min']}, max: {gripper_stat['max']}")
            action_normalizers.append(get_range_normalizer_from_stat(gripper_stat))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)
        print(f"✓ Created action normalizer")
        print(f"{'='*80}\n")

        # obs
        print(f"{'='*80}")
        print(f"[Normalizer Creation] Creating low-dim observation normalizers...")
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            
            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
                norm_type = "range"
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
                norm_type = "range"
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
                norm_type = "identity"
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
                norm_type = "range"
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer
            
            # Print key statistics
            if 'eef_pos' in key or 'gripper' in key:
                print(f"{key} ({norm_type}): min={stat['min']}, max={stat['max']}")

        print(f"✓ Created {len(self.lowdim_keys)} low-dim normalizers")
        print(f"{'='*80}\n")

        # image
        print(f"{'='*80}")
        print(f"[Normalizer Creation] RGB images use identity normalizer (no normalization)")
        print(f"RGB keys: {self.rgb_keys}")
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        print(f"{'='*80}\n")
        
        # tactile points normalization
        if self.pointnet_use_normalization and self.tactile_points_keys:
            print(f"\n{'='*80}")
            print(f"[Point Cloud Normalization] Starting point cloud normalization...")
            print(f"[Point Cloud Normalization] Number of tactile point cloud keys: {len(self.tactile_points_keys)}")
            print(f"[Point Cloud Normalization] Keys: {self.tactile_points_keys}")
            print(f"[Point Cloud Normalization] Norm method: {self.pointnet_norm_method}")
            print(f"[Point Cloud Normalization] Per-sensor normalization: {self.pointnet_norm_per_sensor}")
            
            if self.pointnet_norm_per_sensor:
                # Separate normalization for each sensor
                for key in self.tactile_points_keys:
                    pc = data_cache[key].reshape(-1, data_cache[key].shape[-2], 3)  # (BT,N,3)
                    non_zero = np.any(pc != 0, axis=-1)                              # (BT,N)
                    pc_filtered = pc[non_zero].reshape(-1, 3)                        # (M,3)
                    
                    print(f"\n[Point Cloud Normalization] Sensor: {key}")
                    print(f"  - Original shape: {data_cache[key].shape}")
                    print(f"  - Reshaped to: {pc.shape}")
                    print(f"  - Non-zero points: {pc_filtered.shape[0]} / {pc.shape[0] * pc.shape[1]}")
                    
                    stat = array_to_stats(pc_filtered)
                    print(f"  - Mean: {stat['mean']}")
                    print(f"  - Std: {stat['std']}")
                    print(f"  - Min: {stat['min']}")
                    print(f"  - Max: {stat['max']}")
                    
                    normalizer[key] = get_pointcloud_normalizer_from_stat(
                        stat=stat,
                        norm_method=self.pointnet_norm_method,
                        center_method=self.pointnet_norm_center_method,
                        scale_method=self.pointnet_norm_scale_method
                    )
                    print(f"  ✓ Created normalizer with method: {self.pointnet_norm_method}")
            else:
                # Shared normalization across all sensors
                print(f"\n[Point Cloud Normalization] Computing shared normalization across all sensors...")
                all_pc = []
                for key in self.tactile_points_keys:
                    pc = data_cache[key].reshape(-1, data_cache[key].shape[-2], 3)
                    non_zero = np.any(pc != 0, axis=-1)
                    pc_filtered = pc[non_zero].reshape(-1, 3)
                    all_pc.append(pc_filtered)
                    
                    print(f"\n[Point Cloud Normalization] Sensor: {key}")
                    print(f"  - Original shape: {data_cache[key].shape}")
                    print(f"  - Reshaped to: {pc.shape}")
                    print(f"  - Non-zero points: {pc_filtered.shape[0]} / {pc.shape[0] * pc.shape[1]}")
                    
                if len(all_pc):
                    pc_all = np.concatenate(all_pc, axis=0)
                    stat = array_to_stats(pc_all)
                    
                    print(f"\n[Point Cloud Normalization] Combined statistics (all sensors):")
                    print(f"  - Total non-zero points: {pc_all.shape[0]}")
                    print(f"  - Mean: {stat['mean']}")
                    print(f"  - Std: {stat['std']}")
                    print(f"  - Min: {stat['min']}")
                    print(f"  - Max: {stat['max']}")
                    
                    shared = get_pointcloud_normalizer_from_stat(
                        stat=stat,
                        norm_method=self.pointnet_norm_method,
                        center_method=self.pointnet_norm_center_method,
                        scale_method=self.pointnet_norm_scale_method
                    )
                    for key in self.tactile_points_keys:
                        normalizer[key] = shared
                    print(f"  ✓ Created shared normalizer with method: {self.pointnet_norm_method}")
                else:
                    # fallback: identity pointcloud normalizer
                    print(f"\n[Point Cloud Normalization] ⚠️  WARNING: No valid point cloud data found!")
                    for key in self.tactile_points_keys:
                        normalizer[key] = get_pointcloud_normalizer_from_stat(
                            stat=array_to_stats(np.zeros((1,3),dtype=np.float32)),
                            norm_method="identity", center_method="centroid", scale_method="max_dist"
                        )
                    print(f"[Point Cloud Normalization] Using identity pointcloud normalizer as fallback")
            print(f"{'='*80}\n")
        else:
            # Use identity pointcloud normalizer (no normalization)
            print(f"\n{'='*80}")
            print(f"[Point Cloud Normalization] Point cloud normalization is DISABLED")
            print(f"[Point Cloud Normalization] Using identity normalizer for {len(self.tactile_points_keys)} sensors")
            for key in self.tactile_points_keys:
                normalizer[key] = get_pointcloud_normalizer_from_stat(
                    stat=array_to_stats(np.zeros((1,3),dtype=np.float32)),
                    norm_method="identity", center_method="centroid", scale_method="max_dist"
                )
            print(f"{'='*80}\n")
            
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.  # normalized
            # T,C,H,W
            del data[key]
        # Handle tactile images (same as RGB processing)
        for key in self.tactile_img_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
            
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # Handle tactile points data (fixed array format)
        for key in self.tactile_points_keys:
            if key in data:
                tactile_data = data[key]  # Shape: (T, num_points, 3) - fixed array format
                
                # Convert numpy array to torch tensor
                obs_dict[key] = torch.from_numpy(tactile_data.astype(np.float32)).contiguous()
                del data[key]
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep( # left hand to right hand
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)   # 9d
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep( # correct
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'], # tag2tcp
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
            
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)
        
        # Convert obs_dict to torch tensors (tactile_points are already tensors)
        torch_obs_dict = {}
        for key, value in obs_dict.items():
            if key in self.tactile_points_keys:
                # tactile_points are already torch tensors
                torch_obs_dict[key] = value
            else:
                torch_obs_dict[key] = torch.from_numpy(value)
        
        torch_data = {
            'obs': torch_obs_dict,
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data
