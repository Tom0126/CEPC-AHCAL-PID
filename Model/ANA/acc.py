import matplotlib.pyplot as plt
import numpy as np


# combination: 98.86%
# pi_path='../ahcal_beamtest/pi+.npy'
# e_path='../ahcal_beamtest/e+.npy'
# mu_path='../ahcal_beamtest/mu+.npy'

def plotACC(combi_path, mu_path, e_path, pi_path, save_path, n_classes=3, proton_path=None):
    combi_acc = np.load(combi_path)
    e_acc = np.load(e_path)
    pi_acc = np.load(pi_path)
    mu_acc = np.load(mu_path)
    lower_limit = np.min([np.min(pi_acc[1]), np.min(e_acc[1]), np.min(mu_acc[1])])
    upper_limit = np.max([np.max(pi_acc[1]), np.max(e_acc[1]), np.max(mu_acc[1])])
    energy_points = sorted([100, 20, 40, 60, 80, 120, 30, 50, 70, 90, 160])
    plt.figure(figsize=(6, 5))

    if n_classes == 4:  # with proton classfication
        proton_acc = np.load(proton_path)
        lower_limit = np.min([lower_limit, np.min(proton_acc[1])])
        upper_limit = np.max([upper_limit, np.max(proton_acc[1])])
        plt.plot(proton_acc[0], proton_acc[1], 'o', color='darkorange', label='proton', markersize=4)
        
    plt.text(20, 96.6, 'CEPC Preliminary', fontsize=15, fontstyle='oblique', fontweight='bold')
    plt.text(20, 96.3, 'AHCAL PID', fontsize=12, fontstyle='normal')
    # combination acc
    plt.plot(np.linspace(20, 160, 10), combi_acc[0] * np.ones(10), linestyle=':', color='black')
    plt.text(100, combi_acc[0], 'Overall Accuracy: {}%'.format(combi_acc[0]))
    # base acc
    plt.plot(np.linspace(20, 160, 10), lower_limit * np.ones(10), linestyle=':', color='blueviolet')
    plt.text(107, lower_limit, 'Lowest Accuracy: {}%'.format(lower_limit), color='blueviolet')
    plt.plot(np.linspace(20, 160, 10), upper_limit * np.ones(10), linestyle=':', color='blueviolet')
    plt.text(108, upper_limit, 'Highest Accuracy: {}%'.format(upper_limit), color='blueviolet')
    #  e+ acc
    plt.plot(e_acc[0], e_acc[1], 'o', color='blue', label='e+', markersize=4)
    #  p+ acc
    plt.plot(pi_acc[0], pi_acc[1], 'o', color='red', label='pion+', markersize=4)
    #  mu+ acc
    plt.plot(mu_acc[0], mu_acc[1], 'o', color='green', label='mu+', markersize=4)

    plt.ylim([np.min([lower_limit, 95]), 100.5])
    plt.xticks(energy_points)
    plt.legend(loc='lower right')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Accuracy [%]')
    plt.savefig(save_path)
    plt.close()
    # plt.show()
