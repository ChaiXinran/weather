from openstl.methods import method_maps
from openstl.methods.evolution_temporal_unet import EvolutionTemporalUNet


def test_temporal_unet_method_registration_aliases_match():
    assert method_maps['evolutiontemporalunet'] is EvolutionTemporalUNet
    assert method_maps['evolution_temporal_unet'] is EvolutionTemporalUNet
