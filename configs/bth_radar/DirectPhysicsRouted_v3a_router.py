"""V3a stage B: freeze candidate experts and train only the router."""

from configs.bth_radar.DirectPhysicsRouted_v3a import *

v3a_stage = 'router'
lr = 1e-4
epoch = 3
