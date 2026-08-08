"""V3a stage C: jointly fine-tune U-Net, motion, decay, and router."""

from configs.bth_radar.DirectPhysicsRouted_v3a import *

v3a_stage = 'joint'
lr = 5e-5
epoch = 8
