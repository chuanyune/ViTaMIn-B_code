from typing import List, Optional, Union, Dict, Callable
import numbers
import copy
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from real_world.uvc_camera import UvcCamera
from real_world.video_recorder import VideoRecorder

class MultiUvcCamera:
    def __init__(self,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_paths: List[str],
            shm_manager: Optional[SharedMemoryManager]=None,
            resolution=(1280,720),
            capture_fps=60,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
            video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
            # tactile point cloud params
            enable_tactile_pc: Optional[Union[bool, List[bool]]]=None,
            fps_num_points: Optional[Union[int, List[int]]]=None,
            tactile_lower_bound: Optional[Union[int, List[int]]]=None,
            verbose=False
        ):
        super().__init__()

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n_cameras = len(dev_video_paths)

        resolution = repeat_to_list(
            resolution, n_cameras, tuple)
        capture_fps = repeat_to_list(
            capture_fps, n_cameras, (int, float))
        cap_buffer_size = repeat_to_list(
            cap_buffer_size, n_cameras, int)
        receive_latency = repeat_to_list(
            receive_latency, n_cameras, (int, float))
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)
        video_recorder = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)
        # Handle tactile point cloud parameters
        enable_tactile_pc = repeat_to_list(
            enable_tactile_pc, n_cameras, bool)
        fps_num_points = repeat_to_list(
            fps_num_points, n_cameras, int)
        tactile_lower_bound = repeat_to_list(
            tactile_lower_bound, n_cameras, int)
        
        cameras = dict()
        for i, path in enumerate(dev_video_paths):
            # Debug: print tactile parameters for each camera
            if enable_tactile_pc[i]:
                print(f"Camera {i} ({path}): enable_tactile_pc={enable_tactile_pc[i]}, "
                      f"fps_num_points={fps_num_points[i]}, tactile_lower_bound={tactile_lower_bound[i]}")
            
            cameras[path] = UvcCamera(
                shm_manager=shm_manager,
                dev_video_path=path,
                resolution=resolution[i],
                capture_fps=capture_fps[i],
                put_fps=put_fps,
                put_downsample=put_downsample,
                get_max_k=get_max_k,
                receive_latency=receive_latency[i],
                cap_buffer_size=cap_buffer_size[i],
                transform=transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorder[i],
                # tactile point cloud params
                enable_tactile_pc=enable_tactile_pc[i] if enable_tactile_pc[i] is not None else False,
                fps_num_points=fps_num_points[i] if fps_num_points[i] is not None else 256,
                tactile_lower_bound=tactile_lower_bound[i] if tactile_lower_bound[i] is not None else 10,
                verbose=verbose
            )

        self.cameras = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def start_recording(self, video_path: Union[str, List[str]], start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
