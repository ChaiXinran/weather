"""Validate an explicit checkpoint without training or full test evaluation."""

import warnings

warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, load_config, update_config


if __name__ == '__main__':
    args = create_parser().parse_args()
    if args.config_file is None or (
            args.ckpt_path is None and not args.motion_eval_only):
        raise ValueError(
            '--config_file and --ckpt_path are required unless '
            '--motion_eval_only is used')
    config = args.__dict__
    config = update_config(
        config, load_config(args.config_file),
        exclude_keys=['method', 'val_batch_size'])
    for attribute, value in default_parser().items():
        if config[attribute] is None:
            config[attribute] = value
    experiment = BaseExperiment(args)
    # In motion-eval mode the method loads only backbone/decoder/motion_head
    # from evolution_motion_checkpoint; source-head stays at initialization.
    experiment.trainer.validate(
        experiment.method, experiment.data,
        ckpt_path=None if args.motion_eval_only else args.ckpt_path)
