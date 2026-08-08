# Copyright (c) CAIRI AI Lab. All rights reserved

from .convlstm import ConvLSTM
from .direct_physics_hybrid import DirectPhysicsHybrid
from .direct_physics_routed import DirectPhysicsRouted
from .evolution_convlstm import EvolutionConvLSTM
from .evolution_temporal_unet import EvolutionTemporalUNet
from .e3dlstm import E3DLSTM
from .mau import MAU
from .mim import MIM
from .phydnet import PhyDNet
from .predrnn import PredRNN
from .predrnnpp import PredRNNpp
from .predrnnv2 import PredRNNv2
from .simvp import SimVP
from .tau import TAU
from .mmvp import MMVP
from .swinlstm import SwinLSTM_D, SwinLSTM_B
from .wast import WaST

method_maps = {
    'convlstm': ConvLSTM,
    'directphysicshybrid': DirectPhysicsHybrid,
    'direct_physics_hybrid': DirectPhysicsHybrid,
    'directphysicsrouted': DirectPhysicsRouted,
    'direct_physics_routed': DirectPhysicsRouted,
    'evolutionconvlstm': EvolutionConvLSTM,
    'evolution_convlstm': EvolutionConvLSTM,
    'evolutiontemporalunet': EvolutionTemporalUNet,
    'evolution_temporal_unet': EvolutionTemporalUNet,
    'e3dlstm': E3DLSTM,
    'mau': MAU,
    'mim': MIM,
    'phydnet': PhyDNet,
    'predrnn': PredRNN,
    'predrnnpp': PredRNNpp,
    'predrnnv2': PredRNNv2,
    'simvp': SimVP,
    'tau': TAU,
    'mmvp': MMVP,
    'swinlstm_d': SwinLSTM_D,
    'swinlstm_b': SwinLSTM_B,
    'swinlstm': SwinLSTM_B,
    'wast': WaST
}

__all__ = [
    'method_maps', 'ConvLSTM', 'DirectPhysicsHybrid', 'DirectPhysicsRouted',
    'EvolutionConvLSTM', 'EvolutionTemporalUNet',
    'E3DLSTM', 'MAU', 'MIM',
    'PredRNN', 'PredRNNpp', 'PredRNNv2', 'PhyDNet', 'SimVP', 'TAU',
    "MMVP", 'SwinLSTM_D', 'SwinLSTM_B', 'WaST'
]
