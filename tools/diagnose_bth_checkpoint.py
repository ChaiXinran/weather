"""Run the full BTH precipitation evaluator on the validation split."""

import warnings

warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser, load_config, update_config


if __name__ == '__main__':
    args = create_parser().parse_args()
    if args.config_file is None or args.ckpt_path is None:
        raise ValueError('--config_file and --ckpt_path are required')
    config = update_config(
        args.__dict__, load_config(args.config_file),
        exclude_keys=['method', 'val_batch_size'])
    for attribute, value in default_parser().items():
        if config[attribute] is None:
            config[attribute] = value

    experiment = BaseExperiment(args)
    # Reuse the ordered test-time evaluator against the validation dataset so
    # event IDs, spatial diagnostics, and sample ordering remain identical.
    experiment.data.test_loader = experiment.data.valid_loader
    experiment.test()
