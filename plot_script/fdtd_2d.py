import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def show_material_space(data_dir: str):
    coord_x = np.loadtxt(os.path.join(data_dir, 'coord_x.dat'))
    coord_y = np.loadtxt(os.path.join(data_dir, 'coord_y.dat'))
    sigma_e_z = np.loadtxt(os.path.join(data_dir, 'sigma_e_z.dat'))

    print(coord_x.shape)
    print(coord_y.shape)
    print(sigma_e_z.shape)
    #set max and min value for color map
    sigma_e_z = np.clip(sigma_e_z, 0, 0.1)

    f, ax = plt.subplots(1, 1, figsize=(8,8))
    f.suptitle('Material Space Sigma E_z')
    [x, y] = np.meshgrid(coord_x, coord_y)
    ax.pcolor(x, y, sigma_e_z, cmap='jet')
    # ax.imshow(sigma_e_z, cmap='jet', extent=[coord_x[0], coord_x[-1], coord_y[0], coord_y[-1]])
    # ax.contourf(x, y, sigma_e_z, 100, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    f.savefig(os.path.join(data_dir, 'material_space.png'))

    tfsf_e_i = np.loadtxt(os.path.join(data_dir, 'tfsf_e_i.dat'))
    f,ax = plt.subplots(1,1, figsize=(8,8))
    line, = ax.plot(tfsf_e_i[0, :])
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.set_title('Ex Line Monitor T: 0 step')

    def update(frame):
        line.set_ydata(tfsf_e_i[frame, :])
        ax.set_title(f'Ex Line Monitor T: {frame} step')
        return line,

    # ani = FuncAnimation(f, update, frames=np.arange(1, tfsf_e_i.shape[0], 8), blit=True)
    # ani.save(os.path.join(data_dir, 'tfsf_e_i.gif'), writer='ffmpeg', fps=10)

    ez_gif_data_dir = os.path.join(data_dir, 'ez')
    ez_files = os.listdir(ez_gif_data_dir)
    ez_files.sort()
    ez_files = [os.path.join(ez_gif_data_dir, file) for file in ez_files]
    ez_data = [np.loadtxt(file) for file in ez_files]
    # ez_data = 10 * np.log10(np.abs(ez_data) + 1e-4)
    f, ax = plt.subplots(1, 1, figsize=(8, 8))
    # 2d
    im = ax.imshow(ez_data[0], cmap='jet', vmin=-1, vmax=1)
    ax.set_title('Ez 2D')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    f.colorbar(im)

    def update(frame):
        im.set_array(ez_data[frame])
        ax.set_title(f'Ez 2D T: {frame}')
        return im,
    ani = FuncAnimation(f, update, frames=np.arange(1, len(ez_data), 1), blit=True)
    ani.save(os.path.join(data_dir, 'ez.gif'), writer='ffmpeg', fps=10)


if __name__ == '__main__':
    show_material_space('tmp')
