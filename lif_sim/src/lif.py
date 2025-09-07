# This is a simple implementation of a leaky integrate-and-fire (LIF) neuron model.
# The model includes methods for updating the membrane potential, checking for spikes,
# and resetting the potential after a spike.

# Import necessary libraries
"""
Main script for simulating a leaky integrate-and-fire (LIF) neuron and plotting results.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
from model import LIFNeuron
from plotting import PlotManager 


def main():
    """Run LIF neuron simulation and plot results."""
    # Time vector
    t = np.linspace(0, 100, 1000)  # 100 ms total, 1000 points

    # External current (step current)
    I_ext = np.zeros_like(t)
    I_ext[200:800] = 80  # Apply a current of 80 units from 20 ms to 80 ms

    # Create LIF neuron instance
    lif_neuron = LIFNeuron()

    # Simulate the neuron
    v_trace = lif_neuron.simulate(I_ext, t)

    # Plot results using PlotManager class
    PlotManager.membrane_potential(
        t,
        v_trace,
        lif_neuron.v_thresh,
        title="Leaky Integrate-and-Fire Neuron Simulation",
    )


if __name__ == "__main__":
    main()
