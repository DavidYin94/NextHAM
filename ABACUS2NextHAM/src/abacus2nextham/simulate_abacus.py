#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from ase.io import write
from constant import PATH_PSEUDO, PATH_ORBITAL, PP, BASIS, ELEMENTS


def write_stru(path_input, ase_atoms, ase_stru_name):
    ase_atoms_elements = ase_atoms.get_chemical_symbols()
    stru_pp = {}
    stru_orbit = {}
    for iele in ase_atoms_elements:
        if(iele in ELEMENTS):
            stru_pp[iele] = PP[iele]
            stru_orbit[iele] = BASIS[iele]
        else:
            raise(f"ERROR: not find {iele} PP or ORBIT\n")
    write(os.path.join(path_input, f'{ase_stru_name}'),
            ase_atoms,
            format='abacus',
            pp=stru_pp,
            basis=stru_orbit
            )
    return 


def get_abacus_scf(path_icase, ase_atoms):
    path_input = os.path.join(path_icase, f'abacus_scf')
    if (not os.path.exists(path_input)):
        os.mkdir(path_input)

    os.chdir(path_input)
    def write_input(path_input, ntype):
        with open(os.path.join(path_input, f'INPUT'), 'w') as f:
            f.write(f'INPUT_PARAMETERS\n')
            f.write(f'suffix                         ABACUS\n')
            f.write(f'stru_file                      STRU\n')
            f.write(f'pseudo_dir                     {PATH_PSEUDO}\n')
            f.write(f'orbital_dir                    {PATH_ORBITAL}\n')
            f.write(f'calculation                    scf\n')
            f.write(f'scf_nmax                       100\n')
            f.write(f'\n')
            f.write(f'ntype                          {ntype}\n')
            f.write(f'nspin                          4\n')
            f.write(f'ecutwfc                        100\n')
            f.write(f'scf_thr                        1e-06\n')
            f.write(f'ks_solver                      genelpa\n')
            f.write(f'basis_type                     lcao\n')
            f.write(f'gamma_only                     0\n')
            f.write(f'smearing_method                gauss\n')
            f.write(f'symmetry                       0\n')
            f.write(f'mixing_type                    broyden\n')
            f.write(f'kspacing                       0.10\n')
            f.write(f'\n')
            f.write(f'out_mul                        1\n')
            f.write(f'out_mat_hs2                    1\n')
            f.write(f'lspinorb                       1\n')
            f.write(f'\n')

    pp = {}
    basis = {}
    for iele in ase_atoms.symbols:
        pp[iele] = PP[iele]
        basis[iele] = BASIS[iele]
    write_input(path_input=path_input,
                ntype=len(set(ase_atoms.get_chemical_symbols())))
    write_stru(path_input, ase_atoms, 'STRU')
    os.chdir(path_icase)
    return path_input


def get_abacus_geths(path_icase, ase_atoms):
    path_input = os.path.join(path_icase, f'abacus_geths')
    if (not os.path.exists(path_input)):
        os.mkdir(path_input)

    os.chdir(path_input)
    def write_input(path_input, ntype):
        with open(os.path.join(path_input, f'INPUT'), 'w') as f:
            f.write(f'INPUT_PARAMETERS\n')
            f.write(f'pseudo_dir                     {PATH_PSEUDO}\n')
            f.write(f'orbital_dir                    {PATH_ORBITAL}\n')
            f.write(f'calculation                    get_hs\n')
            f.write(f'\n')
            f.write(f'ntype                          {ntype}\n')
            f.write(f'nspin                          4\n')
            f.write(f'ecutwfc                        100\n')
            f.write(f'scf_thr                        1e-06\n')
            f.write(f'ks_solver                      genelpa\n')
            f.write(f'basis_type                     lcao\n')
            f.write(f'gamma_only                     0\n')
            f.write(f'smearing_method                gauss\n')
            f.write(f'symmetry                       0\n')
            f.write(f'mixing_type                    broyden\n')
            f.write(f'out_mat_hs2                    1\n')
            f.write(f'kspacing                       0.10\n')
            f.write(f'lspinorb                       1\n')
            f.write(f'\n')

    pp = {}
    basis = {}
    for iele in ase_atoms.symbols:
        pp[iele] = PP[iele]
        basis[iele] = BASIS[iele]
    write_input(path_input=path_input,
                ntype=len(set(ase_atoms.get_chemical_symbols())))
    write_stru(path_input, ase_atoms, 'STRU')
    os.chdir(path_icase)
    return path_input