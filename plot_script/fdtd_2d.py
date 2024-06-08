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

    print("MAX: ", np.max(sigma_e_z))
    print("MIN: ", np.min(sigma_e_z))
    #set max and min value for color map
    sigma_e_z = np.clip(sigma_e_z, 0, 0.1)

    f, ax = plt.subplots(1, 1, figsize=(8,8))
    f.suptitle('Material Space Sigma E_z')
    [x, y] = np.meshgrid(coord_x, coord_y)
    ax.pcolor(x, y, sigma_e_z, cmap='jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    f.savefig(os.path.join(data_dir, 'material_space.png'))

if __name__ == '__main__':
    show_material_space('tmp')
