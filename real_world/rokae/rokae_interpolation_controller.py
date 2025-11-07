import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) + "/.."
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import enum
import numpy as np
import multiprocessing as mp
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from real_world.rokae.rokae_interface import RokaeInterface

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from utils.precise_sleep import precise_wait
from utils.pose_util import pose_to_mat, mat_to_pose


class CustomError(Exception):
    def __init__(self, message):
        self.message = message

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class RokaeInterpolationController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        launch_timeout=3,
        verbose = False,
        receive_latency=0.0,
        arm_name=None,
        max_pos_speed=0.25, # 5% of max speed
        max_rot_speed=0.16, # 5% of max speed
        frequency = 100,   # TODO: modify
        soft_real_time=False,
        get_max_k=None): 
        super().__init__(name="RrankaPositionalController")

        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.receive_latency = receive_latency
        self.frequency = frequency
        self.soft_real_time = soft_real_time

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=512
        )

        example = dict()

        example['ee_pose'] = np.zeros(6)
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
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
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.arm_name = arm_name
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RokaePositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        # print("$$$$$$$$$$\n$$$$$$$$$$\n$$$$$$$$$$\n$$$$$$$$$$\n$$$$$$$$$$")
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)    # Block until ready_event is set or timeout
        assert self.is_alive()                        # Ensure controller process or thread is started and active
    
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
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

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
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        robot = RokaeInterface(arm_name=self.arm_name)

        try:
            if self.verbose:
                print(f"[RokaePositionalController] Connect to robot")

            dt = 1. / self.frequency
            curr_pose = mat_to_pose(pose_to_mat(robot.get_obs))

            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                cam_pose = pose_interp(t_now)
                ee_pose = mat_to_pose(pose_to_mat(cam_pose))
                if not robot.check():
                    raise CustomError("Rokae Check Error")

                robot.execute(ee_pose, Delta=False)

                # update robot state
                state = dict()
                state['ee_pose'] = mat_to_pose(pose_to_mat(robot.get_obs))
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        print("*************************************************\n\n\n\n")
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RokaePositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        # print("*************************************************")
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break
                    

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        except Exception as e:
            print(f"Exception occurred: {e}")
            raise
        finally:
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.stop(reason = None)
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[RokaePositionalController] Disconnected from robot")