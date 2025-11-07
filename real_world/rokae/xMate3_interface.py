import time
import rospy
import numpy as np
from copy import deepcopy
from threading import Lock
from impedance_control.msg import JointControlCommand, CartesianControlCommand, ImpedanceRobotState

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Protocol, Iterable
from typing import Literal, Tuple, Any
if TYPE_CHECKING:
    Size = TypeVar("Size", bound=str, covariant=True)
    ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
    class Shaped_array(Iterable, Protocol[Size, ScalarType]):
        def __getitem__(self, __i: Any) -> Any: ...
else:
    Shaped_array = Tuple

XMATE_DOF: int = 7

FAUCET_INIT_JOINT_POSE: Shaped_array[Literal["(7,)"], np.float64] = [-0.5, -0.143, 0.0, 1.2, 0.0, 1.57, 1.57]
DEFAULT_JOINT_IMPEDANCE: Shaped_array[Literal["(7,)"], np.float64] = [200.0, 200.0, 200.0, 200.0, 50.0, 50.0, 50.0]

class _control_interface(ABC):

    _lock: Lock
    _command_pub: rospy.Publisher

    _fake: bool

    __arm_state: ImpedanceRobotState

    _name: str

    @property
    def arm_state(self) -> ImpedanceRobotState:
        with self._lock:
            return deepcopy(self.__arm_state)

    def __init__(self, name:str, fake: bool) -> None:
        self._lock = Lock()
        self._fake = fake
        self._name = name
        self.__arm_state = ImpedanceRobotState()

        rospy.Subscriber("impedance_robot_state_" + self._name, ImpedanceRobotState, self.__xMate_listener, queue_size=1)
        self._initialize_robotcontrol()

    @abstractmethod
    def _initialize_robotcontrol(self) -> None: ...

    def __xMate_listener(self, msg: ImpedanceRobotState) -> None:
        with self._lock:
            self.__arm_state = msg

    @abstractmethod
    def exec_arm(self, target) -> None: ...

class Joint_control_interface(_control_interface):

    stiffness: Shaped_array[Literal["(7,)"], np.float64]

    def __init__(self, stiffness: Shaped_array[Literal["(7,)"], np.float64], name: str = 'A', fake: bool = False) -> None:
        super().__init__(**{"name": name, "fake": fake})
        self.stiffness = stiffness

    def _initialize_robotcontrol(self) -> None:
        self._command_pub = rospy.Publisher('joint_control_command_' + self._name, JointControlCommand, queue_size=1)
        time.sleep(1)
        if not self._fake:
            assert self._command_pub.get_num_connections() > 0, "No subscriber for joint_control_command, controller may not be started or control mode mismatch"

    def exec_arm(self, target: Shaped_array[Literal["(7,)"], np.float64]) -> None:
        self._command_pub.publish(JointControlCommand(target, self.stiffness, (20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0), False))

class Cartesian_control_interface(_control_interface):

    stiffness: Shaped_array[Literal["(6,)"], np.float64]

    def __init__(self, stiffness: Shaped_array[Literal["(6,)"], np.float64], name: str = 'A', fake: bool = False) -> None:
        super().__init__(name, **{"fake": fake})
        self.stiffness = stiffness
        self._name = name

    def _initialize_robotcontrol(self) -> None:
        self._command_pub = rospy.Publisher('cartesian_control_command_' + self._name, CartesianControlCommand, queue_size=1)
        time.sleep(1)
        if not self._fake:
            assert self._command_pub.get_num_connections() > 0, "No subscriber for cartesian_control_command, controller may not be started or control mode mismatch"

    def exec_arm(self, target: Shaped_array[Literal["(16,)"], np.float64]) -> None:
        self._command_pub.publish(CartesianControlCommand(target, self.stiffness, tuple(100 for i in range(6)), False))

if __name__ == "__main__":
    np.printoptions(4)
    rospy.init_node("XMate_control_interface_demo")

    import random
    def r(rng: float) -> float: return (random.random() - 0.5) * 2 * rng
    jci = Joint_control_interface(DEFAULT_JOINT_IMPEDANCE)
    jci.exec_arm(FAUCET_INIT_JOINT_POSE)
    time.sleep(2)
    for i in range(10):
        jci.exec_arm([-0.5 + r(0.05), -0.143 + r(0.05), 0 + r(0.05), 1.2 + r(0.1), 0 + r(0.1), 1.57 + r(0.1), 1.57 + r(0.1)])
        print(jci.arm_state)
        time.sleep(2)

    import math
    cci = Cartesian_control_interface((1800, 1800, 1800, 100, 100, 100))
    cci.exec_arm([0.478810, 0.764405, 0.431771, 0.25, 0.877919, -0.416613, -0.235993, -0.27, -0.000513, 0.492055, -0.870564, 0.60, 0.000000, 0.000000, 0.000000, 1.000000])
    time.sleep(2)
    for deg in range(0, 1080, 5):
        cci.exec_arm([0.478810, 0.764405, 0.431771, 0.25, 0.877919, -0.416613, -0.235993, -0.27 + 0.05 * math.sin(deg / 180 * math.pi), -0.000513, 0.492055, -0.870564, 0.55 + 0.05 * math.cos(deg / 180 * math.pi), 0.000000, 0.000000, 0.000000, 1.000000])
        print(cci.arm_state)
        time.sleep(0.06)
