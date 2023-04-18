import os
import numpy as np

#import mdtraj as mdj

import torch
from torch import Tensor
from typing import Tuple, Optional, NamedTuple

from torch import nn
from torch.jit import Final
from typing import List


def calculateCOG(pos: Tensor) -> Tensor:
    return torch.mean(pos, dim=0)


def translateCoordinates(pos: Tensor, t: Tensor) -> Tensor:
    return pos - t


def shiftbyCOG(pos: Tensor) -> Tensor:
    cog = calculateCOG(pos)
    return translateCoordinates(pos, cog)


class Quaternion(nn.Module):
    """Finds the optimal rotation between two molecules using quaternion.
    The details of this method can be found on
    https://onlinelibrary.wiley.com/doi/10.1002/jcc.20110.
    """

    normquat: Final[Tuple[float, float, float, float]]
    # group_idxs: Final[List[int]]
    # fittingGroup_idxs: Final[List[int]]
    refpositions_file: Final[str]
    refpos: List[List[float]]
    S_eigvec: Tensor
    S_eigval: Tensor
    C: Tensor
    S: Tensor
    q: Tensor

    def __init__(self, group_idxs: List[int], refpositions: List[List[float]],
                 fittingGroup_idxs: Optional[List[int]] = None, 
                 normquat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)):
        """Constructor for the Quaternion model
        """
        super().__init__()
        self.is_trainable = False
        self.normquat = normquat
        self.group_idxs = group_idxs
        self.fittingGroup_idxs = fittingGroup_idxs
        #self.refpositions_file = refpositions_file
        self.S = torch.zeros(4, 4)
        self.S_eigval = torch.zeros(4)
        self.S_eigvec = torch.zeros(4, 4)
        self.q = torch.zeros(4)
        self.refpos = refpositions #mdj.load_pdb(self.refpositions_file).xyz[0].tolist()
        # self.register_buffer()

    def build_correlation_matrix(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        # TODO: check the dimentions
        # check the type and device
        return torch.matmul(pos1.T, pos2)

    def calculate_overlap_matrix(self, C: Tensor) -> Tensor:

        S = torch.zeros(4, 4, dtype=C.dtype, device=C.device)

        S[0][0] = C[0][0] + C[1][1] + C[2][2]
        S[1][1] = C[0][0] - C[1][1] - C[2][2]
        S[2][2] = - C[0][0] + C[1][1] - C[2][2]
        S[3][3] = - C[0][0] - C[1][1] + C[2][2]
        S[0][1] = C[1][2] - C[2][1]
        S[0][2] = - C[0][2] + C[2][0]
        S[0][3] = C[0][1] - C[1][0]
        S[1][2] = C[0][1] + C[1][0]
        S[1][3] = C[0][2] + C[2][0]
        S[2][3] = C[1][2] + C[2][1]
        S[1][0] = S[0][1]
        S[2][0] = S[0][2]
        S[2][1] = S[1][2]
        S[3][0] = S[0][3]
        S[3][1] = S[1][3]
        S[3][2] = S[2][3]

        self.S = S
        return S

    def diagonalize_matrix(self):

        U, S, V = torch.linalg.svd(self.S)
        self.S_eigval = S**2 / (S**2).sum()
        self.S_eigvec = V

        # convert complex values to real
        self.S_eigval = torch.real(self.S_eigval)
        self.S_eigvec = torch.real(self.S_eigvec)


    def getQfromEigenvecs(self, idx: int) -> Tensor:
        normquat = torch.tensor(self.normquat, dtype=self.S.dtype, device=self.S.device)
        if torch.matmul(self.S_eigvec[idx, :], normquat) < 0:
            return -1 * self.S_eigvec[idx, :]
        else:
            return self.S_eigvec[idx, :]

    def calc_optimal_rotation(self, pos1: Tensor, pos2: Tensor) -> Tensor:

        C = self.build_correlation_matrix(pos1, pos2)
        self.calculate_overlap_matrix(C)
        self.diagonalize_matrix()
        self.q = self.getQfromEigenvecs(0)
        return self.q

    def rotateCoordinates(self, q: Tensor, pos: Tensor) -> Tensor:
        return torch.stack([self.quaternionRotate(q, row) for row in pos])

    def quaternionInvert(self, q: Tensor):

        return torch.tensor([q[0], -q[1], -q[2], -q[3]],
                            dtype=q.dtype,
                            device=q.device)

    def quaternionRotate(self, q: Tensor, vec: Tensor):
        q0 = q[0]
        vq = torch.tensor([q[1], q[2], q[3]], dtype=q.dtype, device=q.device)
        a = torch.cross(vq, vec) + q0 * vec
        b = torch.cross(vq, a)
        return b + b + vec

    def forward(self, pos: Tensor):

        if self.fittingGroup_idxs is None:
            # Cenetr groupA pos
            centered_pos = shiftbyCOG(pos[self.group_idxs])

            # Center refpos group A
            refpos = torch.tensor(self.refpos, dtype=pos.dtype, device=pos.device)
            centerded_refpos = shiftbyCOG(refpos)
            rot_q = self.calc_optimal_rotation(centerded_refpos, centered_pos)

        else:
            # second group used for fitting 
            # center ref pos
            refpos = torch.tensor(self.refpos, dtype=pos.dtype, device=pos.device)
            centered_refpos = shiftbyCOG(refpos[self.group_idxs])
            centered_refpos_fitgroup = shiftbyCOG(refpos[self.fittingGroup_idxs])

            # calc COG of the fitting group
            fitgroup_COG = calculateCOG(pos[self.fittingGroup_idxs])
            centered_pos = translateCoordinates(pos[self.group_idxs], fitgroup_COG)
            centered_pos_fitgroup = translateCoordinates(pos[self.fittingGroup_idxs], fitgroup_COG)
            fit_rot_q = self.calc_optimal_rotation(centered_pos_fitgroup, centered_refpos_fitgroup)

            # rotate both groups
            rotated_pos = self.rotateCoordinates(fit_rot_q, centered_pos)
            #rotsted_pos_fitgroup = self.rotateCoordinates(fit_rot_q, centered_pos_fitgroup)

            # find optimal rotation between aligned group atoms 
            rot_q = self.calc_optimal_rotation(centered_refpos, rotated_pos)

        return rot_q[0]
