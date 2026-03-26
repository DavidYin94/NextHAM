#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import subprocess
from constant import SLURM_NTASKS, SLURM_PART, ABACUS_SCF,ABACUS_SCF_ENV, ABACUS_GETHS, ABACUS_GETHS_ENV


def submit_slurm_job(script_path):
    try:
        result = subprocess.run(
            ['sbatch', script_path],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.strip().split('\n'):
            if line.startswith('Submitted batch job'):
                job_id = int(line.split()[-1])
                print(f"Job submitted successfully. Job ID: {job_id}")
                return job_id

        print("Warning: No Job ID found in sbatch output.")
        return None

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr.strip()}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None
    

def write_slurm(path_input, icase, if_geths):
    with open(os.path.join(path_input, f'submit.slurm'), 'w') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'#SBATCH -J STRU_{icase}\n')
        f.write(f'#SBATCH -p {SLURM_PART}\n')
        f.write(f'#SBATCH -N 1\n')
        f.write(f'#SBATCH --ntasks={SLURM_NTASKS}\n')
        f.write(f'#SBATCH -o job%j.log\n')
        f.write(f'#SBATCH -e job%j.err\n')
        f.write(f'\n\n')
        f.write(f'echo "Start time: `date`"\n')
        if (if_geths):
            f.write(f'source {ABACUS_GETHS_ENV}\n')
            f.write(f'OMP_NUM_THREADS=1 mpirun -np 1 {ABACUS_GETHS}\n')
        else:
            f.write(f'source {ABACUS_SCF_ENV}\n')
            f.write(f'OMP_NUM_THREADS=1 mpirun -np {SLURM_NTASKS} {ABACUS_SCF}\n')
        f.write(f'echo "End time: `date`"\n')
    return f'submit.slurm'
