import matplotlib.pyplot as plt


class PlotManager:
    """
    Handles plotting for neuron simulation results.
    """

    @staticmethod
    def membrane_potential(
        t, v_trace, threshold, title='Leaky Integrate-and-Fire Neuron Simulation'
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(t, v_trace, label='Membrane Potential (mV)')
        plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
        plt.title(title)
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.legend()
        plt.grid()
        plt.show()
