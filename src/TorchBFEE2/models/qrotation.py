import os
import numpy as np

import mdtraj as mdj

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


class Qrotation(nn.Module):
    """Finds the optimal rotation between two molecules using quaternion.
    The details of this method can be found on
    https://onlinelibrary.wiley.com/doi/10.1002/jcc.20110.
    """

    normquat: Final[Tuple[float, float, float, float]]
    # groupA_idxs: Final[List[int]]
    # groupB_idxs: Final[List[int]]
    refpos_pdb_path: Final[str]
    refpos: List[List[float]]
    S_eigvec: Tensor
    S_eigval: Tensor
    C: Tensor
    S: Tensor
    q: Tensor

    def __init__(self, groupA_idxs: List[int], refpos_pdb_path: str,
                 groupB_idxs: Optional[List[int]] = None, normquat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)):
        """Constructor for the qrotation model
        """
        super().__init__()
        self.is_trainable = False
        self.normquat = normquat
        self.groupA_idxs = groupA_idxs
        self.groupB_idxs = groupB_idxs
        self.refpos_pdb_path = refpos_pdb_path
        self.S = torch.zeros(4, 4)
        self.S_eigval = torch.zeros(4)
        self.S_eigvec = torch.zeros(4, 4)
        self.q = torch.zeros(4)
        self.refpos = mdj.load_pdb(self.refpos_pdb_path).xyz[0].tolist()
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

        normquat = torch.tensor(self.normquat, dtype=self.S.dtype, device=self.S.device)
        for idx, dotprodcut in enumerate(torch.matmul(self.S_eigvec, normquat)):
            if dotprodcut < 0:
                self.S_eigvec[idx] *= -1

    def getQfromEigenvecs(self, idx: int) -> Tensor:
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

        # Cenetr groupA pos

        centered_pos1 = shiftbyCOG(pos[self.groupA_idxs])

        # Center refpos group A
        refpos = torch.tensor(self.refpos, dtype=pos.dtype, device=pos.device)
        centerded_refpos1 = shiftbyCOG(refpos[self.groupA_idxs])

        return self.calc_optimal_rotation(centerded_refpos1, centered_pos1)
