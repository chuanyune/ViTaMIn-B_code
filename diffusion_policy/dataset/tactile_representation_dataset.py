from typing import Dict
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    BaseSequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.dataset.real_data_conversion import real_data_to_replay_buffer

class TactileRepresentationDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            resized_image_shape=(128, 160),
        ):

        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store,
                store=zarr.MemoryStore()
            )
        # # list all the files in the folder
        # files = os.listdir(dataset_path)
        # assert len(files) > 0, f"No files found in {dataset_path}"
        # files = [os.path.join(dataset_path, f) for f in files]
        # replay_buffer = None
        # for dataset_file in files:
        #     # Open the current Zarr zip file
        #     with zarr.ZipStore(dataset_file, mode='r') as zip_store:
        #         if replay_buffer is None:
        #             # Initialize the replay buffer from the first file
        #             replay_buffer = ReplayBuffer.copy_from_store(
        #                 src_store=zip_store,
        #                 store=zarr.MemoryStore()
        #             )
        #             print(replay_buffer.n_episodes)
        #         else:
        #             temp_buffer = ReplayBuffer.copy_from_store(
        #                 src_store=zip_store,
        #                 store=zarr.MemoryStore()
        #             )
        #             temp_buffer.get_compressors()
        #             # data = temp_buffer.get_chunks()
        #             replay_buffer.extend(temp_buffer.data)  # Replace with the appropriate method
        #             print(replay_buffer.n_episodes)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        sampler = BaseSequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.val_mask = val_mask
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.resized_image_shape = resized_image_shape
        # Dummy value
        self.horizon = 1
        self.n_obs_steps = 1

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = BaseSequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()
        for key in self.rgb_keys:
            # convert uint8 image to float32
            # T, H, W, C
            # save image
            obs_dict[key] = data[key][T_slice].astype(np.float32) / 255.
            del data[key]
        del data
        # def resize_image(image):
        #     H,W = self.resized_image_shape
        #     resized_image = cv2.resize(image, (W,H))
        #     return resized_image
        #
        # for key in obs_dict.keys():
        #     obs_dict[key] = resize_image(obs_dict[key][0])
        #
        # obs_dict['tactile'] = np.concatenate(
        #     [obs_dict['camera0_left_tactile'], obs_dict['camera0_right_tactile']], axis=0
        # )
        # # in H, W, C format
        # res = {
        #     'tactile': obs_dict['tactile'],
        # }
        return obs_dict


