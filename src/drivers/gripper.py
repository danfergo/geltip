from narnia.core import ROSActuator, ROSTopicPublisher, ROSTopicSubscriber, Driver, RequestAction


@Driver()
class GripperDriver(ROSActuator):

    def __init__(self):
        self.STATUS = None
        self.GOAL = None
        self.default_speed = 255
        self.cmd = {
            'rSP': self.default_speed,
            'rFR': 30
        }
        super().__init__()

    @ROSTopicPublisher('/Robotiq2FGripperRobotOutput')
    def send_cmd(self, msg):
        self.cmd = {**self.cmd, **msg}
        return self.cmd

    @ROSTopicSubscriber('/Robotiq2FGripperRobotInput')
    def on_status(self, status):
        self.STATUS = status


    def reset(self):
        self.send_cmd({'rACT': 0})
        return RequestAction(self, lambda ts, t: t - ts > 0.25 and self.STATUS['gACT'] == 0)

    def set_opening(self, opening, speed=None):
        pos = int(255 - min(max(0, opening * 255), 255))

        s = self.default_speed if speed is None else int(255 - min(max(0, speed * 255), 255))
        cmd = {'rACT': 1, 'rGTO': 1, 'rPR': pos, 'rSP': s}
        self.send_cmd(cmd)
        return RequestAction(self,
                             lambda ts, t: t - ts > 0.25 and
                                           self.STATUS['gSTA'] == 3 and
                                           self.STATUS['gOBJ'] != 0 and
                                           self.STATUS['gPR'] == pos)

    def get_opening(self):
        mn = 3
        mx = 229
        pos = self.STATUS['gPO']
        pos = max(pos - mn, 0) / (mx - mn)
        return 1 - pos

    def open(self):
        return self.set_opening(1)

    def close(self, **kwargs):
        return self.set_opening(0, **kwargs)

    def stop(self):
        self.send_cmd({'rPR': 255})
        return RequestAction(self, lambda ts, t: t - ts > 0.1)