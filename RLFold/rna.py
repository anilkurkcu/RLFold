import gym
from gym import spaces
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import pyglet
import os
import time
import rmsd
import pymol
import random

pymol.finish_launching(['pymol', '-qc'])
cmd = pymol.cmd

class RNA(gym.Env):

    def __init__(self):
        super().__init__()

        self.info = None
        self.state = None
        self.viewer = None
        self.atom_coords = None
        self.snap_dir = ''
        self.snap_count = 0
        self.folder_des = '/mnt/sod2-project/csb4/wgs/anil/backup/database/dataset_train'
        self.folder_tst = '/mnt/sod2-project/csb4/wgs/anil/backup/database/dataset_test'

    def set_rna(self, pdb_file):
        cmd.reinitialize()
        cmd.load(pdb_file)
        print(pdb_file)
        while len(cmd.get_chains()) > 1:
            elim = cmd.get_chains()[-1]
            comm = 'chain ' + elim
            cmd.remove(comm)
        self.atom_coords = cmd.get_coords()
        
        rid = cmd.get_names()[0]
        seq1 = ''.join(cmd.get_fastastr().split('\n')[1:])
        seq = seq1.split('>')[0]
        L = len(seq)
        self.L = L

        self.info = {'id': rid, 'seq': seq, 'len': L, 'rmsd': None}

        self.snap_dir = f'{rid}-3D'

        chain = cmd.get_chains()[0]
        model = cmd.get_model(f'chain {chain}')
        self.info['start_resi_n'] = model.atom[0].resi_number
        
        perturbation_amount = 0.1 # Initial perturbation for starting structure.
        angle_choice = 'all' # Angle to perturb.

        self.init_per_angle(angle_choice, perturbation_amount)

        self.initial_torsion_angles = self._torsion_angles()

    def _torsion_angles(self): # Read torsion angles of the current structure.
        torsion_angles = np.array([])
        for i in range(self.info['start_resi_n'], self.info['len'] + self.info['start_resi_n']):
 
            if i != self.info['start_resi_n']:
                torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i-1}/O3'", f"{i}/P", f"{i}/O5'", f"{i}/C5'")) # Alpha

                torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i}/P", f"{i}/O5'", f"{i}/C5'", f"{i}/C4'")) # Beta

            torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i}/O5'", f"{i}/C5'", f"{i}/C4'", f"{i}/C3'")) # Gamma

            torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i}/C5'", f"{i}/C4'", f"{i}/C3'", f"{i}/O3'")) # Delta

            if i != (self.info['len'] + self.info['start_resi_n'] - 1):
                torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i}/C4'", f"{i}/C3'", f"{i}/O3'", f"{i+1}/P")) # Epsilon

                torsion_angles = np.append(torsion_angles, cmd.get_dihedral(f"{i}/C3'", f"{i}/O3'", f"{i+1}/P", f"{i+1}/O5'")) # Zeta

        torsion_angles_new = np.pad(torsion_angles, (0, 116 - torsion_angles.shape[0]))

        return torsion_angles_new

    def _nuc_to_vec(self): # One-hot encoding for input sequence.

        sequence = self.info['seq']
        seq_vector = np.array([])

        for seq in sequence:
            if seq == 'A':
                seq_vector = np.append(seq_vector, [1,0,0,0])
            elif seq == 'U':
                seq_vector = np.append(seq_vector, [0,1,0,0])
            elif seq == 'G':
                seq_vector = np.append(seq_vector, [0,0,1,0])
            elif seq == 'C':
                seq_vector = np.append(seq_vector, [0,0,0,1])

        seq_vector_new = np.pad(seq_vector, (0, 80 - seq_vector.shape[0]))

        return seq_vector_new

    def _set_residue_torsion(self, idx, angles): # Set structure according to applied perturbations.
        alpha, beta, gamma, epsilon, zeta = angles
        i = self.info['start_resi_n'] + idx
        L = self.info['len']
        assert 0 <= idx < L, 'Index error'

        if idx != 0:
            # alpha
            if alpha is not None:
                alpha += cmd.get_dihedral(f"{i-1}/O3'", f"{i}/P", f"{i}/O5'", f"{i}/C5'")
                cmd.set_dihedral(f"{i-1}/O3'", f"{i}/P", f"{i}/O5'", f"{i}/C5'", alpha)

            # beta
            if beta is not None:
                beta += cmd.get_dihedral(f"{i}/P", f"{i}/O5'", f"{i}/C5'", f"{i}/C4'")
                cmd.set_dihedral(f"{i}/P", f"{i}/O5'", f"{i}/C5'", f"{i}/C4'", beta)

        # gamma
        if gamma is not None:
            gamma += cmd.get_dihedral(f"{i}/O5'", f"{i}/C5'", f"{i}/C4'", f"{i}/C3'")
            cmd.set_dihedral(f"{i}/O5'", f"{i}/C5'", f"{i}/C4'", f"{i}/C3'", gamma)

        # delta
        #if delta is not None:
            #delta += cmd.get_dihedral(f"{i}/C5'", f"{i}/C4'", f"{i}/C3'", f"{i}/O3'")
            #cmd.set_dihedral(f"{i}/C5'", f"{i}/C4'", f"{i}/C3'", f"{i}/O3'", delta)

        if idx != L - 1:
            # epsilon
            if epsilon is not None:
                epsilon += cmd.get_dihedral(f"{i}/C4'", f"{i}/C3'", f"{i}/O3'", f"{i+1}/P")
                cmd.set_dihedral(f"{i}/C4'", f"{i}/C3'", f"{i}/O3'", f"{i+1}/P", epsilon)

            # zeta
            if zeta is not None:
                zeta += cmd.get_dihedral(f"{i}/C3'", f"{i}/O3'", f"{i+1}/P", f"{i+1}/O5'")
                cmd.set_dihedral(f"{i}/C3'", f"{i}/O3'", f"{i+1}/P", f"{i+1}/O5'", zeta)

        # chi: dont consider side chain at now

        cmd.unpick()

    def init_per_angle(self, angle_choice, perturbation_amount): # Decide upon a certain angle and apply a certain amount of perturbation to all residues.
        for i in range(self.info['len']):
            angles = [np.random.uniform(-perturbation_amount, perturbation_amount, 1) if angle_choice == 'all' else None, np.random.uniform(-perturbation_amount, perturbation_amount, 1) if angle_choice == 'all' else None, np.random.uniform(-perturbation_amount, perturbation_amount, 1) if angle_choice == 'all' else None, np.random.uniform(-perturbation_amount, perturbation_amount, 1) if angle_choice == 'all' else None, np.random.uniform(-perturbation_amount, perturbation_amount, 1) if angle_choice == 'all' else None]
            self._set_residue_torsion(i, angles)

    def reset(self):
        cmd.reinitialize()
        random_sequence = random.choice(os.listdir(self.folder_des))
        self.set_rna(os.path.join(self.folder_des, random_sequence))

        self.state_new = self._torsion_angles()
        nuc2vec = self._nuc_to_vec()
        self.state_new = np.append(self.state_new / 180, nuc2vec)

        return self.state_new

    def reset_test(self):
        cmd.reinitialize()
        random_sequence = random.choice(os.listdir(self.folder_tst))
        self.set_rna(os.path.join(self.folder_tst, random_sequence))

        self.state_new = self._torsion_angles()
        nuc2vec = self._nuc_to_vec()
        self.state_new = np.append(self.state_new / 180, nuc2vec)

        return self.state_new

    def step(self, action):
        for i in range(self.L):
            residue, angles = i, [action[i], action[i+20], action[i+40], action[i+60], action[i+80]]
            self._set_residue_torsion(residue, angles)

        self.state = cmd.get_coords()
        reward_rmsd = rmsd.kabsch_rmsd(np.round(self.state.copy(), 2), np.round(self.atom_coords.copy(), 2), translate=True)
        self.info['rmsd'] = -(reward_rmsd)
        self.state_new = self._torsion_angles()
        nuc2vec = self._nuc_to_vec()
        self.state_new2 = np.append(self.state_new / 180, nuc2vec)

        reward_angle = reward_rmsd

        reward_angle_scaled = -(reward_angle) * 100
        done = reward_angle_scaled > 0
        return self.state_new2, reward_angle_scaled, done, self.info

    def step_test(self, action):
        for i in range(self.L):
            residue, angles = i, [action[i], action[i+20], action[i+40], action[i+60], action[i+80]]
            self._set_residue_torsion(residue, angles)

        self.state = cmd.get_coords()
        reward_rmsd = rmsd.kabsch_rmsd(np.round(self.state.copy(), 2), np.round(self.atom_coords.copy(), 2), translate=True)
        self.info['rmsd'] = -(reward_rmsd)
        self.state_new = self._torsion_angles()
        nuc2vec = self._nuc_to_vec()
        self.state_new2 = np.append(self.state_new / 180, nuc2vec)

        reward_angle = 0

        reward_angle_scaled = -(reward_angle) * 100
        done = reward_angle_scaled > 0
        return self.state_new2, reward_angle_scaled, done, self.info
