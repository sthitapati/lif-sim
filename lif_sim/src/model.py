"""
Model classes for lif-sim project.
"""

import numpy as np


class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=-65.0, v_reset=-70.0, v_thresh=-50.0, r=1.0):
        self.tau = tau
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r = r
        self.v = v_rest
        self.spike_times = []

    def update(self, I, dt):
        dv = (-(self.v - self.v_rest) + self.r * I) / self.tau
        self.v += dv * dt
        if self.v >= self.v_thresh:
            self.spike()

    def spike(self):
        self.spike_times.append(len(self.spike_times))
        self.v = self.v_reset

    def simulate(self, I_ext, t):
        dt = t[1] - t[0]
        v_trace = np.zeros_like(t)
        for i in range(len(t)):
            self.update(I_ext[i], dt)
            v_trace[i] = self.v
        return v_trace
