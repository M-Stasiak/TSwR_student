import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kp = 55
        self.Kd = 10

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = x[:2]
        q_dot = x[2:]

        M = self.model.M(x)
        C = self.model.C(x)

        # 2.
        # v = q_r_ddot
        # u = M @ v + C @ q_dot

        # 8.
        v = q_r_ddot + self.Kd * (q_r_dot - q_dot) + self.Kp * (q_r - q)
        u = M @ v + C @ q_dot

        return u
