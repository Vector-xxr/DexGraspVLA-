"""智元 OmniPicker 夹爪：485 + Modbus RTU 驱动。说明见同目录 README.md。

所有命令都在 SO101 目录下跑（和 atom_robot 一样，`python -m` 从当前目录找包）。
"""
from .config import OmniPickerConfig
from .gripper import OmniPickerGripper, SafetyStop
from .protocol import FAULTS, STATES, ModbusError, OmniPickerBus

__all__ = ["OmniPickerConfig", "OmniPickerGripper", "SafetyStop",
           "OmniPickerBus", "ModbusError", "FAULTS", "STATES"]
