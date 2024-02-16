#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 17:21
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : pid_jobs.py
# @Software: PyCharm

import os
import glob
def main(ep_list, pid_bash_path):



    for i, ep in enumerate(ep_list):

        os.system('cp {} {}'.format(pid_bash_path, pid_bash_path.replace('.sh', '_{}.sh'.format(i))))

        os.system('sed -i 26cep\={} {}'.format(ep,pid_bash_path.replace('.sh', '_{}.sh'.format(i))))
        os.system('sbatch {}'.format(pid_bash_path.replace('.sh', '_{}.sh'.format(i))))

def main_calculate(path_format, pid_bash_path):



    for i, _ in enumerate(glob.glob(path_format)):
        print(_)
        os.system('cp {} {}'.format(pid_bash_path, pid_bash_path.replace('.sh', '_{}.sh'.format(i))))

        os.system("sed -i 25cstring\='{}' {}".format(_,pid_bash_path.replace('.sh', '_{}.sh'.format(i))))
        os.system('sbatch {}'.format(pid_bash_path.replace('.sh', '_{}.sh'.format(i))))


if __name__ == '__main__':

    # TODO ======================== check ========================
    # ep_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]  # On pi+ beam in 2022.
    # main(ep_list=ep_list, pid_bash_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/main.sh')

    # ep_list = [10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]  # On pi+ beam in 2023.
    #
    # main(ep_list=ep_list, pid_bash_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/main.sh')

    main_calculate(path_format='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/ihep_mc/mc_0720_e_pi_block_1_1/*',
                   pid_bash_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/ShowerTopology/main.sh')
    pass
