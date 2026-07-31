"""Validate an explicit checkpoint without training or full test evaluation."""

import warnings

warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, load_config, update_config


if __name__ == '__main__':
    args = create_parser().parse_args()
    if args.config_file is None or args.ckpt_path is None:
        raise ValueError('--config_file and --ckpt_path are required')
    config = args.__dict__
    config = update_config(
        config, load_config(args.config_file),
        exclude_keys=['method', 'val_batch_size'])
    for attribute, value in default_parser().items():
        if config[attribute] is None:
            config[attribute] = value
    experiment = BaseExperiment(args)
    experiment.trainer.validate(
        experiment.method, experiment.data, ckpt_path=args.ckpt_path)
