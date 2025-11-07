from typing import Optional
import numpy as np
import random
import numba
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray, sequence_length: int,
        episode_mask: np.ndarray,
        pad_before: int = 0, pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def get_val_mask(n_episodes, val_ratio, seed=0):
    """
    Create validation mask for splitting training and validation sets
    
    Args:
        n_episodes: Total number of trajectories
        val_ratio: Validation set ratio (0~1)
        seed: Random seed
    Returns:
        Boolean array, True indicates trajectory is used for validation
    """
    # Create mask array with all False
    val_mask = np.zeros(n_episodes, dtype=bool)
    
    # If validation ratio <= 0, return all False mask directly
    if val_ratio <= 0:
        return val_mask

    # Calculate number of validation trajectories
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    # min(..., n_episodes-1): Ensure at least one trajectory is left for training
    # max(1, ...): Ensure at least one trajectory for validation
    
    # Create random number generator
    rng = np.random.default_rng(seed=seed)
    
    # Randomly select validation trajectory indices
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    
    # Mark selected indices as True in mask
    val_mask[val_idxs] = True
    return val_mask

def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class BaseSequenceSampler:
    def __init__(self,
                 replay_buffer: ReplayBuffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=dict(),
                 episode_mask: Optional[np.ndarray] = None,
                 ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert (sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends,
                                     sequence_length=sequence_length,
                                     pad_before=pad_before,
                                     pad_after=pad_after,
                                     episode_mask=episode_mask
                                     )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:],
                                 fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx + k_data]
                except Exception as e:
                    import pdb;
                    pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_down_sample_steps: dict,
        tactile_img_keys: list = None,  # Tactile image keys parameter
        tactile_points_keys: list = None,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None
    ):
        
        episode_ends = replay_buffer.episode_ends[:]  # Create a copy of data
        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08  # TODO
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    """
                    # Consider sampling 16 actions:
                    # - 1st action: current_idx + 0
                    # - 2nd action: current_idx + 3
                    # - 3rd action: current_idx + 6
                    # ...
                    # - 16th action: current_idx + (16-1)*3

                    # Total steps needed = (16-1)*3 + 1
                    # - (16-1)*3: covers 15 intervals
                    # - +1: includes starting point
                    """
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))
        
        # load low_dim to memory and keep rgb/tactile as compressed zarr array
        self.original_replay_buffer = replay_buffer  # Keep reference to original replay_buffer
        self.replay_buffer = dict()  # Used to store processed low-dimensional data
        self.num_robot = 0
        
        # Handle tactile_points_keys if provided
        if tactile_points_keys is None:
            tactile_points_keys = []
        
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1

            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)] # key[:-4] = 'robot0_eef_pos'  # Remove '_abs' suffix
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        # Load tactile points data as-is (fixed tensor arrays)
        for key in tactile_points_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.tactile_img_keys = tactile_img_keys if tactile_img_keys is not None else []  # Save tactile image keys
        self.tactile_points_keys = tactile_points_keys
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.tactile_img_keys + self.lowdim_keys + self.tactile_points_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.tactile_img_keys + self.lowdim_keys + self.tactile_points_keys
        
        # observation
        for key in obs_keys:
            # Get data from correct buffer
            if key in self.replay_buffer:
                # Low-dimensional data is already preprocessed and stored in memory
                input_arr = self.replay_buffer[key]
            else:
                # RGB and tactile image data read from original replay_buffer
                if key not in self.original_replay_buffer:
                    print(f'ERROR: Key {key} not found in original_replay_buffer!')
                    print(f'Available keys: {list(self.original_replay_buffer.keys())}')
                    continue
                input_arr = self.original_replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            
            # simplified sampling without latency compensation
            num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
            slice_start = current_idx - (num_valid - 1) * this_downsample_steps

            output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
            assert output.shape[0] == num_valid
            
            # solve padding
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
                
            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply