import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) + '/..' + '/..'
# print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
print(ROOT_DIR)

import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real_world.pgi.pgi_interface import PGIInterface
from utils.precise_sleep import precise_wait
from utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class PGIController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            ser,
            calibration_path,
            frequency=30,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=1,
            receive_latency=0.0,
            verbose=False,
            timeout = 0.02,
            move_max_speed = 200.0
            ):
        super().__init__(name="PGIController")
        self.ser = ser
        self.calibration_path = calibration_path
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.timeout = timeout
        self.move_max_speed = move_max_speed
        self.idddx = 0

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PGIController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)
        # print("******************************************")


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        try:
            with PGIInterface(serial_name = self.ser, calibration_file_path=self.calibration_path, timeout = self.timeout) as pgi:
                curr_pos, _ = pgi.get_obs()
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos,0,0,0,0,0]]
                )
                
                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    # command grippers
                    t_now = time.monotonic()
                    dt = 1 / self.frequency
                    t_target = t_now
                    target_pos = pose_interp(t_target)[0]
                    target_vel = abs(target_pos - pose_interp(t_target - dt)[0]) / dt
                    pgi.set_velocity(target_vel)
                    pgi.set_pos(target_pos)
                    pos, vel = pgi.get_obs()

                    # get state from robot
                    state = {
                        'gripper_position': pos,
                        'gripper_velocity': vel,
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency
                    }
                    self.ring_buffer.put(state)

                    # fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0
                    # print(n_cmd)

                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']
                        
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False                # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            # print('enter')
                            target_pos = command['target_pos']
                            # print("cmd", target_pos)
                            target_time = command['target_time']
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.move_max_speed,
                                max_rot_speed=self.move_max_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                            # print(pose_interp.poses)
                            # print(pose_interp.times)
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break
                        
                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        print("\n\n\n*******gripper's ready_event is set*******\n\n\n")
                        self.ready_event.set()   # Set an event to occur
                    iter_idx += 1
                    
                    # Limit frequency here
                    dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
                    # print("gripper on")
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[PGIController] Disconnected from robot: {self.ser}")