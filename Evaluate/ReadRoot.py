import uproot
import numpy as np


def readRootFileCell(path):
    file = uproot.open(path)

    # A TTree File
    simulation = file['T']

    # Get data->numpy
    data = simulation.arrays(library="np")

    # e,x,y,z
    hcal_energy = data['hcal_celle']
    x = data['hcal_cellx']
    y = data['hcal_celly']
    z = data['hcal_cellz']

    return hcal_energy, x, y, z


if __name__ == '__main__':
    file_path = '/lustre/collider/songsiyuan/CEPC/PID/e+/resultFile/hcal_e+_50GeV_2cm_10k.root'
    hcal_energy, x, y, z=readRootFileCell(file_path)
    # print('x_min: {}, x_max: {}'.format(np.min((x[99])/4),np.max((x[99]+34)/4)))
    # print('y_min: {}, y_max: {}'.format(np.min((y[99] + 34) / 4), np.max((y[99] + 34) / 4)))
    # print('z_min: {}, z_max: {}'.format(np.min((z[99] - 301.5) / 25), np.max((z[99]-301.5) / 25)))
    print('x_min: {}, x_max: {}'.format(np.min((x[99] +340)/ 40), np.max((x[99] + 340) / 40)))
    print('y_min: {}, y_max: {}'.format(np.min((y[99] + 340) / 40), np.max((y[99] + 340) / 40)))
    print('z_min: {}, z_max: {}'.format(np.min((z[99] - 301.5) / 25), np.max((z[99]-301.5) / 25)))
    print((z[99] - 301.5) / 25)