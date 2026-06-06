import numpy as np

from models.manipulator_model import ManiuplatorModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([
            [3 * p[0], 0],
            [0, 3 * p[1]],
            [3 * p[0] ** 2, 0],
            [0, 3 * p[1] ** 2],
            [p[0] ** 3, 0],
            [0, p[1] ** 3]
        ])
        W = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])

        M = self.model.M(x)
        C = self.model.C(x)
        M_inv = np.linalg.inv(M)

        A = np.block([
            [np.zeros((2, 2)), np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), -M_inv @ C, np.eye(2)],
            [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]
        ])

        B = np.concatenate([
            np.zeros((2, 2)),
            M_inv,
            np.zeros((2, 2))
        ], axis=0)

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q = x[:2]

        M = self.model.M(x)
        C = self.model.C(x)

        z = self.eso.get_state()
        q_hat = z[:2]
        q_hat_dot = z[2:4]
        f_hat = z[4:]

        v = q_d_ddot + self.Kd @ (q_d_dot - q_hat_dot) + self.Kp @ (q_d - q)
        
        u = M @ (v - f_hat) + C @ q_hat_dot

        self.update_params(q_hat, q_hat_dot)
        self.eso.update(q, u)

        return u
