import numpy as np


class lqr(object):


    def __init__(self, state, target, horizon=10):


        self.A = np.array([
            [1., 0., 0., 0.02, 0., 0.],
            [0., 1., 0., 0., 0.02, 0.],
            [0., 0., 1., 0., 0., 0.02],
            [0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]
        ])
        self.B = np.array([
            [0.002, 0., 0.],
            [0., -0.002, 0.],
            [0., 0., 0.0045],
            [0.196, 0., 0.],
            [0., -0.196, 0.],
            [0., 0., 0.4545]
        ])

        self.Q = np.diag((1., 1., 5., 0.8, 0.8, 0.8))
        # self.Q = sparse.diags([10., 10., 5., 1., 1., 1.])
        self.F = self.Q
        self.R = np.diag((8., 8., 8.))

        self.Rinv = np.linalg.inv(self.R)
        self.time_step = 1/50
        self.horizon = horizon
        self.sat_val = 6.0
        self.active = 1
        self.final_cost = 0

        self.target_state = target
        self.state = state


    def get_control_gains(self): # finite horizon LQR
        P = [None] * self.horizon
        P[-1] = self.F.copy()
        K = [None] * (self.horizon)
        r = [None] * (self.horizon)
        K[-1] = self.Rinv.dot(self.B.T.dot(P[-1]))
        r[-1] = self.F.dot(self.target_state*0.0)

        for i in reversed(range(1, self.horizon)):
            PB = np.dot(P[i], self.B)
            BP = np.dot(self.B.T, P[i])
            PBRB = np.dot(PB, np.dot(self.Rinv, self.B.T))
            Pdot = - (np.dot(self.A.T, P[i]) + np.dot(P[i], self.A) - np.dot(np.dot(PB, self.Rinv), BP) + self.Q)
            rdot = -(self.A.T.dot(r[i]) - self.Q.dot(self.target_state) - PBRB.dot(r[i]))
            P[i-1] = P[i] - Pdot*self.time_step
            K[i-1] = self.Rinv.dot(self.B.T.dot(P[i]))
            r[i-1] = r[i] - rdot*self.time_step
        return K, r


    def solve(self):
        K,r = self.get_control_gains()
        ref = -self.Rinv.dot(self.B.T).dot(r[0])
        # print("K[0]: ", K[0])
        # print("Dstate: ", state-self.target_state)
        # return np.clip(-K[0].dot(state-self.target_state), self.sat_val[:, 0], self.sat_val[:, 1]) # only the first is applied (MPC)
        return -K[0].dot(self.state - self.target_state)
    
    def get_linearization_from_trajectory(self, trajectory):
        K,_ = self.get_control_gains() # list with length 20, each element is tuple(4,18)
        return [-k for k in K]
