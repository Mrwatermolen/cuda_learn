import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def plot_gpu_fdtd_1d(data_dir: str):
    ex_line_monitor = np.loadtxt(os.path.join(data_dir, 'ex_line_monitor.dat'))
    ex_reflect_monitor = np.loadtxt(
        os.path.join(data_dir, 'ex_reflect_monitor.dat'))
    ex_transmit_monitor = np.loadtxt(
        os.path.join(data_dir, 'ex_transmit_monitor.dat'))
    incident_wave = np.loadtxt(os.path.join(data_dir, 'incident_wave.dat'))
    time = np.loadtxt(os.path.join(data_dir, 'time.dat'))

    print(ex_line_monitor.shape)
    print(ex_reflect_monitor.shape)
    print(ex_transmit_monitor.shape)
    print(incident_wave.shape)

    # plot reflect, transmit, incident
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle('Reflect, Transmit, Incident')
    ax[0].plot(time * 1e9, ex_reflect_monitor, label='Reflect')
    ax[0].plot(time * 1e9, ex_transmit_monitor, label='Transmit')
    ax[0].plot(time * 1e9, incident_wave, label='Incident')
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].grid()

    reflect_freq_domain = np.fft.fft(ex_reflect_monitor)
    transmit_freq_domain = np.fft.fft(ex_transmit_monitor)
    incident_freq_domain = np.fft.fft(incident_wave)
    dt = time[1] - time[0]
    freq = np.fft.fftfreq(len(time), dt)

    f_min = 0
    f_max = 0.05 / dt
    print(f_min, f_max/1e9)
    index = np.where((freq >= f_min) & (freq <= f_max))

    ax[1].plot(freq[index]/1e9, np.abs(reflect_freq_domain)
               [index], label='Reflect')
    ax[1].plot(freq[index]/1e9, np.abs(transmit_freq_domain)
               [index], label='Transmit')
    ax[1].plot(freq[index]/1e9, np.abs(incident_freq_domain)
               [index], label='Incident')
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    ax[1].grid()

    f.savefig(os.path.join(data_dir, 'reflect_transmit_incident.png'))

    # plot  reflect and transmit coefficient
    reflect_coefficient = np.abs(
        reflect_freq_domain) / np.abs(incident_freq_domain)
    transmit_coefficient = np.abs(
        transmit_freq_domain) / np.abs(incident_freq_domain)
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(freq[index]/1e9, reflect_coefficient[index], label='Reflect FDTD')
    ax.plot(freq[index]/1e9, transmit_coefficient[index],
            label='Transmit FDTD')
    ax.scatter(freq[index]/1e9, abs(1 - 2) / 3.0 *
            np.ones_like(freq[index]), label='Reflect Analytical', marker='o', color='red')
    ax.scatter(freq[index]/1e9, 2 / (1 + 2) * np.ones_like(freq[index]),
            label='Transmit Analytical', marker='*', color='green')

    ax.set_title('Reflect and Transmit Coefficient')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid()
    ax.set_ylim(0, 1)
    f.savefig(os.path.join(data_dir, 'reflect_transmit_coefficient.png'))

    # plot gif
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    line, = ax.plot(ex_line_monitor[0, :])
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.set_title('time = 0 ns')
    slab = 60
    ax.add_patch(plt.Rectangle((0, -1), slab, 2, fill=None, edgecolor='red'))

    def update(frame):
        line.set_ydata(ex_line_monitor[frame, :])
        ax.set_title(f'time = {time[frame]*1e9:.2f} ns')
        return line,

    ani = FuncAnimation(
        f, update, frames=ex_line_monitor.shape[0], blit=True, interval=50)
    ani.save(os.path.join(data_dir, 'ex_line_monitor.gif'), fps=30)


if __name__ == '__main__':
    plot_gpu_fdtd_1d(os.path.join('..', 'tmp'))
