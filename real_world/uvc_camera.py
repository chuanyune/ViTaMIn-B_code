from typing import Optional, Callable, Dict
import enum
import time
import cv2
import numpy as np
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from utils.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from real_world.video_recorder import VideoRecorder

def farthest_point_sampling(points, num_samples):
    if len(points) == 0:
        return np.array([])
    
    points = np.array(points)
    if len(points) <= num_samples:
        return points
    
    # Initialize
    N = len(points)
    selected_indices = []
    distances = np.full(N, np.inf)
    
    # Randomly select first point
    first_idx = np.random.randint(0, N)
    selected_indices.append(first_idx)
    
    # Update distances
    distances = np.minimum(distances, np.linalg.norm(points - points[first_idx], axis=1))
    
    # Iteratively select remaining points
    for _ in range(num_samples - 1):
        # Select point farthest from selected points
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)
        
        # Update minimum distances to selected points
        new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, new_distances)
    
    return points[selected_indices]

def get_tactile_point_cloud(frame, fps_num_points=256, lower_bound=None, calibration_state=None, cam_name=None):
    if calibration_state is None:
        calibration_state = {
            'is_calibrated': False,
            'x_center_err': 0.0,
            'y_center_err': 0.0
        }
    
    start_time = time.time()

    K = np.asmatrix(
        [[506.36, 0, 309.94],
         [0,  506.54, 239.48],
         [0, 0, 1]]
    )
    
    point_cloud = []
    
    lower_bound = lower_bound
    higher_bound = 255

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 50)
    _, bin_frame = cv2.threshold(gray_frame, lower_bound, higher_bound, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Contour Filter
    contours_large = []
    for contour in contours:
        if len(contour) > 500:
            contours_large.append(contour)
    
    contours_cutted = []
    for contour in contours_large:
        contours_cutted.append([point for point in contour if 10 < point[0][1] < 470])
    
    contours_filtered = []
    smooth_len = 5
    for contour in contours_cutted:
        contours_filtered.append([])
        for i in range(0, len(contour) - smooth_len):
            point_avg = np.zeros((1,2))
            for j in range(0, smooth_len):
                point_avg += contour[i+j]
            point_avg /= smooth_len
            
            if np.linalg.norm(point_avg - contour[i]) < 3:
                contours_filtered[-1].append(point_avg.astype(np.int32))
            
        contours_filtered[-1] = np.array(contours_filtered[-1])
    
    # Calibrate the center on first frame
    if not calibration_state['is_calibrated']:

        v0, u0 = K[0,2], K[1,2]
        fv, fu = K[0,0], K[1,1]
        
        # Calculate frame center pixel coordinates as projection on imaging plane
        u_center = 0
        v_center = 0
        point_num = 0
        for contour in contours_filtered:
            for point_ori in contour:
                u = point_ori[0][1]
                v = point_ori[0][0]
                u_center += u
                v_center += v
                point_num += 1
            print("We are now calculating",cam_name, "len of contour=", len(contour))
        
        if point_num > 0:
            u_center /= point_num
            v_center /= point_num

            # Calculate frame center system offset relative to optical center
            u_center_err = u_center - u0
            v_center_err = v_center - v0

            print("We are now calculating",cam_name, "u_center_err=", u_center_err, "v_center_err=", v_center_err)

            # Since frame center and optical center xOy planes coincide, calculate actual offset by intersecting with z=30 plane
            z_beneath = 40
            calibration_state['x_center_err'] = u_center_err * z_beneath/fu
            calibration_state['y_center_err'] = v_center_err * z_beneath/fv
            calibration_state['is_calibrated'] = True

            print(f"{cam_name} sensor calibrated: x_err={calibration_state['x_center_err']:.2f}, y_err={calibration_state['y_center_err']:.2f}")

    # Use calibrated values
    x_center_err = calibration_state['x_center_err']
    y_center_err = calibration_state['y_center_err']

    # Calculate Point Clouds
    v0, u0 = K[0,2], K[1,2]
    fv, fu = K[0,0], K[1,1]

    for contour in contours_filtered:
        # Side plane equation in frame center system from CAD parameters
        # Rubber bottom is 30mm from camera, reference point A(15,16.5,30)
        nx, ny, nz = 0, 33, 4
        Ax, Ay, Az = 15, 16.5, 30
        
        # Transform center system equation to optical system
        nx1, ny1, nz1 = nx + x_center_err, ny + y_center_err, nz
        Ax1, Ay1, Az1 = Ax + x_center_err, Ay + y_center_err, Az
        b1 = Ax1 * nx1 + Ay1 * ny1 + Az1 * nz1
        
        nx2, ny2, nz2 = nx + x_center_err, - ny + y_center_err, nz
        Ax2, Ay2, Az2 = Ax + x_center_err, - Ay + y_center_err, Az
        b2 = Ax2 * nx2 + Ay2 * ny2 + Az2 * nz2
        
        for point_ori in contour:
            u = point_ori[0][1] - u0
            v = point_ori[0][0] - v0
            
            # Initialize variables to avoid undefined behavior
            x, y, z = 0, 0, 0
            
            if v > 0:
                z = b1/(nx1/fu * u + ny1/fv * v + nz1)
                x = z/fu * u
                y = z/fv * v
                if not (abs(z) > 63):
                    point_cloud.append([x, y, z])

            elif v < 0:
                z = b2/(nx2/fu * u + ny2/fv * v + nz2)
                x = z/fu * u
                y = z/fv * v
                if not (abs(z) > 63):
                    point_cloud.append([x, y, z])

            else:
                # Skip this point when v == 0
                continue

    # Apply FPS sampling to ensure fixed point count
    if len(point_cloud) > 0:
        sampled_points = farthest_point_sampling(point_cloud, fps_num_points)
        # If insufficient points, pad with zeros
        if len(sampled_points) < fps_num_points:
            padding = np.zeros((fps_num_points - len(sampled_points), 3))
            sampled_points = np.vstack([sampled_points, padding])
        elif len(sampled_points) > fps_num_points:
            sampled_points = sampled_points[:fps_num_points]
        result = sampled_points
    else:
        # No contact, fill with zeros
        result = np.zeros((fps_num_points, 3))
    
    return result


class Command(enum.Enum):
    RESTART_PUT = 0
    START_RECORDING = 1
    STOP_RECORDING = 2

class UvcCamera(mp.Process):
    """
    Call diffusion_policy.common.usb_util.reset_all_elgato_devices
    if you are using Elgato capture cards.
    Required to workaround firmware bugs.
    """
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes
    
    def __init__(
            self,
            shm_manager: SharedMemoryManager,
            # v4l2 device file path
            # e.g. /dev/video0
            # or /dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB320216MTR-video-index0
            dev_video_path,
            resolution=(1280, 720),
            capture_fps=60,
            put_fps=None,
            put_downsample=True,
            get_max_k=30,
            receive_latency=0.0,
            cap_buffer_size=1,
            num_threads=2,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            # tactile point cloud params
            enable_tactile_pc=False,
            fps_num_points=256,
            tactile_lower_bound=10,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        
        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = {
            'color': np.empty(
                shape=shape+(3,), dtype=np.uint8)
        }
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0
        
        # Add tactile point cloud data if enabled
        if enable_tactile_pc:
            examples['tactile_points'] = np.zeros((fps_num_points, 3), dtype=np.float32)

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=1024
        )

        # create video recorder
        if video_recorder is None:
            # default to nvenc GPU encoder
            video_recorder = VideoRecorder.create_hevc_nvenc(
                shm_manager=shm_manager,
                fps=capture_fps, 
                input_pix_fmt='bgr24', 
                bit_rate=6000*1000)
        assert video_recorder.fps == capture_fps

        # copied variables
        self.shm_manager = shm_manager
        self.dev_video_path = dev_video_path
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.receive_latency = receive_latency
        self.cap_buffer_size = cap_buffer_size
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None
        self.num_threads = num_threads
        # tactile point cloud params
        self.enable_tactile_pc = enable_tactile_pc
        self.fps_num_points = fps_num_points
        self.tactile_lower_bound = tactile_lower_bound
        # Create independent calibration state for each camera instance
        self.tactile_calibration_state = {
            'is_calibrated': False,
            'x_center_err': 0.0,
            'y_center_err': 0.0
        }

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        shape = self.resolution[::-1]
        data_example = np.empty(shape=shape+(3,), dtype=np.uint8)
        self.video_recorder.start(
            shm_manager=self.shm_manager, 
            data_example=data_example)
        # must start video recorder first to create share memories
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.video_recorder.stop()
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
        self.video_recorder.start_wait()
    
    def end_wait(self):
        self.join()
        self.video_recorder.end_wait()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)


    def start_recording(self, video_path: str, start_time: float=-1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(self.num_threads)
        cv2.setNumThreads(self.num_threads)

        # open VideoCapture
        cap = cv2.VideoCapture(self.dev_video_path, cv2.CAP_V4L2)
        
        try:
            # set resolution and fps
            w, h = self.resolution
            fps = self.capture_fps
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            # set fps
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.cap_buffer_size)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set(cv2.CAP_PROP_EXPOSURE, 100)

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            # reuse frame buffer
            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                ts = time.time()
                ret = cap.grab()
                assert ret

                if self.enable_tactile_pc:
                    if not self.tactile_calibration_state['is_calibrated']:
                        print(f"{self.dev_video_path} haven't cali yet! Waiting for cam...")
                        for i in range(150):
                            ret, frame = cap.read()
                            time.sleep(0.03)
                        print(f"{self.dev_video_path} goes on!")

                ret, frame = cap.retrieve()
                t_recv = time.time()
                assert ret

                if self.enable_tactile_pc:
                    if not self.tactile_calibration_state['is_calibrated']:
                        num = self.dev_video_path[10]
                        cv2.imwrite(f"./real_world/cali_check/{num}_cali_frame.png", frame)

                mt_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                t_cap = mt_cap - time.monotonic() + time.time()
                t_cal = t_recv - self.receive_latency

                data = dict()
                data['camera_receive_timestamp'] = t_recv
                data['camera_capture_timestamp'] = t_cap
                data['color'] = frame
                
                # Add tactile point cloud computation if enabled
                if self.enable_tactile_pc:
                    # Debug: print parameters for first few frames
                    if iter_idx < 3:
                        print(f"UvcCamera {self.dev_video_path}: frame {iter_idx}, "
                              f"fps_num_points={self.fps_num_points}, lower_bound={self.tactile_lower_bound}")
                    
                    # Compute tactile point cloud in camera thread for parallel processing
                    tactile_pc = get_tactile_point_cloud(
                        frame, 
                        fps_num_points=self.fps_num_points, 
                        lower_bound=self.tactile_lower_bound,
                        calibration_state=self.tactile_calibration_state,
                        cam_name=self.dev_video_path
                    )
                    # Store as fixed shape array for SharedMemoryRingBuffer compatibility
                    data['tactile_points'] = tactile_pc.astype(np.float32)
                
                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[t_cal],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = t_cal
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((t_cal - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = t_cal
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()    

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[UvcCamera {self.dev_video_path}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start_recording(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop_recording()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            # When everything done, release the capture
            cap.release()

