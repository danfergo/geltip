import asyncio

import narnia
from math import pi
from constants.positions import Positions
from drivers.arm import ArmDriver
from drivers.gripper import GripperDriver

gripper = GripperDriver()
arm = ArmDriver()

async def main():
    while True:
        await asyncio.sleep(1)
        pos = arm.tcp_position()
        print(pos)
    return

    for i in range(10):
        #await arm.move_abs(Positions.start())
        await arm.move_abs(Positions.object())
        pos1 = [0.52, 0.15, 0.20, pi, 0, pi/2]
        pos2 = [0.52, 0.15, 0.05, pi, 0, pi / 2]
        pos3 = [0.52, 0, 0.05, pi, 0, pi / 2]
        await arm.move_abs(pos1)
        await arm.move_abs(pos2)
        await arm.move_abs(pos3)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

