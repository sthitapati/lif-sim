# This is a simple implementation of a leaky integrate-and-fire (LIF) neuron model.
# The model includes methods for updating the membrane potential, checking for spikes,
# and resetting the potential after a spike.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks


class LIFNeuron:
    def __init__(self, tau=20.0, v_rest=-65.0, v_reset=-70.0, v_thresh=-50.0, r=1.0):
        self.tau = tau  # Membrane time constant (ms)
        self.v_rest = v_rest  # Resting potential (mV)
        self.v_reset = v_reset  # Reset potential after spike (mV)
        self.v_thresh = v_thresh  # Spike threshold (mV)
        self.r = r  # Membrane resistance (MΩ)
        self.v = v_rest  # Initial membrane potential (mV)
        self.spike_times = []  # List to record spike times

    def update(self, I, dt):
        """Update the membrane potential based on input current I and time step dt."""
        dv = (-(self.v - self.v_rest) + self.r * I) / self.tau
        self.v += dv * dt

        if self.v >= self.v_thresh:
            self.spike()

    def spike(self):
        """Handle the spike event."""
        self.spike_times.append(
            len(self.spike_times)
        )  # Record spike time (in arbitrary units)
        self.v = self.v_reset  # Reset membrane potential

    def simulate(self, I_ext, t):
        """Simulate the neuron over time with external current I_ext."""
        dt = t[1] - t[0]
        v_trace = np.zeros_like(t)

        for i in range(len(t)):
            self.update(I_ext[i], dt)
            v_trace[i] = self.v

        return v_trace


# Example usage
if __name__ == "__main__":
    # Time vector
    t = np.linspace(0, 100, 1000)  # 100 ms total, 1000 points

    # External current (step current)
    I_ext = np.zeros_like(t)
    I_ext[200:800] = 80  # Apply a current of 1.5 nA from 20 ms to 80 ms

    # Create LIF neuron instance
    lif_neuron = LIFNeuron()

    # Simulate the neuron
    v_trace = lif_neuron.simulate(I_ext, t)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(t, v_trace, label='Membrane Potential (mV)')
    plt.axhline(lif_neuron.v_thresh, color='r', linestyle='--', label='Threshold')
    plt.title('Leaky Integrate-and-Fire Neuron Simulation')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.grid()
    plt.show()
