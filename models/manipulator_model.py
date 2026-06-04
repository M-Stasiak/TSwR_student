import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.7, r3=0.05):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0

        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4

        self.m3 = m3
        self.r3 = r3

        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2

        # 1.
        self.alpha = self.m1 * self.d1 ** 2 + self.I_1 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) + self.I_2
        self.beta = self.m2 * self.l1 * self.d2
        self.gamma = self.m2 * self.d2 ** 2 + self.I_2

        # 4.
        self.alpha += self.m3 * (self.l1 ** 2 + self.l2 ** 2) + self.I_3
        self.beta += self.m3 * self.l1 * self.l2
        self.gamma += self.m3 * self.l2 ** 2 + self.I_3

    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        m_11 = self.alpha + 2 * self.beta * np.cos(q2)
        m_12 = self.gamma + self.beta * np.cos(q2)
        m_21 = m_12
        m_22 = self.gamma

        matrix = np.array([[m_11, m_12], [m_21, m_22]])
        return matrix

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        c_11 = -self.beta * np.sin(q2) * q2_dot
        c_12 = -self.beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = self.beta * np.sin(q2) * q1_dot
        c_22 = 0

        matrix = np.array([[c_11, c_12], [c_21, c_22]])
        return matrix
