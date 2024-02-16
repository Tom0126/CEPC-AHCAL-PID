import uproot
from ReadRoot import *
import argparse


def makeDatasets(file_path, save_path):
    '''
    1: inout: root file
    2: output: numpy array NCHW (,40,18,18)
    '''
    # read raw root file
    hcal_energy, x, y, z = readRootFileCell(file_path)
    num_events = len(hcal_energy)
    assert num_events == len(x)
    assert num_events == len(y)
    assert num_events == len(z)
    # NHWC
    depoits = np.zeros((num_events, 18, 18, 40))
    for i in range(num_events):
        energies_ = hcal_energy[i]
        x_ = ((x[i] + 340) / 40).astype(int)
        y_ = ((y[i] + 340) / 40).astype(int)
        z_ = ((z[i] - 301.5) / 25).astype(int)
        num_events_ = len(energies_)
        assert num_events_ == len(x_)
        assert num_events_ == len(y_)
        assert num_events_ == len(z_)
        for j in range(num_events_):
            depoits[i, x_[j], y_[j], z_[j]] += energies_[j]
    # NCHW
    #depoits = np.transpose(depoits, (0, 3, 1, 2))
    np.save(save_path, depoits)


if __name__ == '__main__':

    # set arg
    parser = argparse.ArgumentParser(
        description='Convert root to HDF5 files')
    parser.add_argument('--infile', '-i', action="store", type=str, required=True,
                        help='input ROOT file')
    parser.add_argument('--outfile', '-o', action="store", type=str, default=None,
                        help='output hdf5 file')
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile


    makeDatasets(infile,outfile)
