#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import subprocess
import time
from ase.io import read
from constant import (
                      PATH_STRU,
                      PATH_SIMULATION,
                      SUBMIT_NUMBER_START,
                      SUBMIT_NUMBER_END
                      )
from simulate_abacus import get_abacus_scf, get_abacus_geths
from scheduler import write_slurm, submit_slurm_job

PATH_RUN = os.getcwd()

def main():
    with open(os.path.join(PATH_RUN, 'running_log.txt'), 'a') as f:
        icase = 0
        all_stru = []
        for root, dirs, files in os.walk(top=PATH_STRU, topdown=True):
            for file in files:
                all_stru.append(file)
        stru_number = len(all_stru)
        print(f'Total Structures: {stru_number}\n')
        for i in range(stru_number):
            if icase > SUBMIT_NUMBER_START and icase < SUBMIT_NUMBER_END:
                file = all_stru[i]
                ase_stru = read(os.path.join(PATH_STRU, file))
                path_icase = os.path.join(PATH_SIMULATION, f'case-{icase}')
                if (not os.path.exists(path_icase)):
                    os.mkdir(path_icase)
                
                for if_geths in [False, True]:
                    if if_geths:
                        path_input = get_abacus_geths(path_icase, ase_stru)
                    else:
                        path_input = get_abacus_scf(path_icase, ase_stru)
                    script_name = write_slurm(path_input, icase, if_geths)
                    script_path = os.path.join(path_input, script_name)
                    os.chdir(path_input)
                    job_id = submit_slurm_job(script_path)
                    os.chdir(PATH_RUN)
                    f.write(f'Case {icase} JobID {job_id} Path {path_icase} File:{file}\n')
            icase += 1
            
            os.chdir(PATH_RUN)

    print(f'Submit Case: {icase}\n')


if __name__ == '__main__':
    main()
