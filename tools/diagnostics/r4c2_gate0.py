"""Run deterministic R4-c2 Gate 0 checks without requiring pytest."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.test_configs.test_bth_evolution_convlstm_config import (
    test_bth_evolution_motion_config_is_motion_only,
    test_single_step_source_config_is_an_explicit_rollback,
)
from tests.test_methods.test_evolution_convlstm import (
    test_pixel_weighted_state_loss_caps_nested_mask_weight,
    test_pixel_weighted_state_loss_uses_one_shared_denominator,
)
from tests.test_models.test_evolution_convlstm import (
    test_bounded_per_step_source_zero_initialization_preserves_r4b,
    test_bounded_source_decoder_parameters_receive_gradient,
    test_bounded_source_single_step_keeps_twenty_flow_head_outputs,
    test_bounded_source_teacher_forcing_uses_true_previous_frame,
    test_evolution_flow_gate_starts_at_configured_scale,
    test_evolution_model_uses_history_only_and_starts_as_persistence,
    test_source_head_zero_initialization_exactly_preserves_motion_only_output,
)
from tests.test_modules.test_evolution_operator import (
    test_backward_warp_positive_dx_moves_content_right_exactly,
    test_bounded_source_has_gradients_on_both_signed_branches,
    test_bounded_source_respects_sink_and_representable_upper_limit,
    test_evolution_operator_returns_diagnostic_fields,
    test_linear_z_warp_preserves_subpixel_peak_better_than_dbz_warp,
    test_no_source_keeps_existing_rain_rate_path_and_diagnostics_empty,
    test_physical_source_rejects_non_rain_rate_operator,
    test_rain_rate_source_is_added_in_mm_per_hour_and_clamped_nonnegative,
    test_stop_gradient_blocks_later_leads_from_earlier_flow,
)


CHECKS = [
    test_backward_warp_positive_dx_moves_content_right_exactly,
    test_evolution_operator_returns_diagnostic_fields,
    test_linear_z_warp_preserves_subpixel_peak_better_than_dbz_warp,
    test_stop_gradient_blocks_later_leads_from_earlier_flow,
    test_rain_rate_source_is_added_in_mm_per_hour_and_clamped_nonnegative,
    test_no_source_keeps_existing_rain_rate_path_and_diagnostics_empty,
    test_physical_source_rejects_non_rain_rate_operator,
    test_bounded_source_respects_sink_and_representable_upper_limit,
    test_bounded_source_has_gradients_on_both_signed_branches,
    test_evolution_model_uses_history_only_and_starts_as_persistence,
    test_evolution_flow_gate_starts_at_configured_scale,
    test_source_head_zero_initialization_exactly_preserves_motion_only_output,
    test_bounded_per_step_source_zero_initialization_preserves_r4b,
    test_bounded_source_teacher_forcing_uses_true_previous_frame,
    test_bounded_source_decoder_parameters_receive_gradient,
    test_bounded_source_single_step_keeps_twenty_flow_head_outputs,
    test_pixel_weighted_state_loss_uses_one_shared_denominator,
    test_pixel_weighted_state_loss_caps_nested_mask_weight,
    test_bth_evolution_motion_config_is_motion_only,
    test_single_step_source_config_is_an_explicit_rollback,
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = []
    for check in CHECKS:
        check()
        results.append({'name': check.__name__, 'status': 'passed'})
        print(f'PASS {check.__name__}')
    report = {
        'status': 'passed',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'checks': results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
