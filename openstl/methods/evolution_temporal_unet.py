from openstl.models import EvolutionTemporalUNet_Model
from .evolution_convlstm import EvolutionPhysicsBase


class EvolutionTemporalUNet(EvolutionPhysicsBase):
    """Temporal U-Net motion backbone with shared physical training logic."""

    def _build_model(self, **args):
        return EvolutionTemporalUNet_Model(self.hparams)
