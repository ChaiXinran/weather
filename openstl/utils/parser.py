# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='Whether to use distributed training (DDP)')
    parser.add_argument('--res_dir', default='work_dirs', type=str)
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str)
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Whether to use Native AMP for mixed precision training (PyTorch=>1.6.0)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='Whether to use torchscripted model')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--fps', action='store_true', default=False,
                        help='Whether to measure inference speed (FPS)')
    parser.add_argument('--test', action='store_true', default=False, help='Only performs testing')
    parser.add_argument(
        '--skip_test_after_train', action='store_true', default=False,
        help='Do not automatically test the in-memory last model after fit')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='whether to set deterministic options for CUDNN backend (reproducable)')

    # dataset parameters
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='Training batch size')
    parser.add_argument('--val_batch_size', '-vb', default=16, type=int, help='Validation batch size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--dataname', '-d', default='mmnist', type=str,
                        choices=['bair', 'mfmnist', 'mmnist', 'mmnist_cifar', 'noisymmnist', 'taxibj', 'human',
                                'kth', 'kth20', 'kth40', 'kitticaltech', 'kinetics', 'kinetics400', 'kinetics600',
                                'weather', 'weather_t2m_5_625', 'weather_mv_4_28_s6_5_625', 'weather_mv_4_4_s6_5_625',
                                'weather_r_5_625', 'weather_uv10_5_625', 'weather_tcc_5_625', 'weather_t2m_1_40625',
                                'weather_r_1_40625', 'weather_uv10_1_40625', 'weather_tcc_1_40625',
                                'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil', 'bth_radar'],
                        help='Dataset name (default: "mmnist")')
    parser.add_argument('--pre_seq_length', default=None, type=int, help='Sequence length before prediction')
    parser.add_argument('--aft_seq_length', default=None, type=int, help='Sequence length after prediction')
    parser.add_argument('--total_length', default=None, type=int, help='Total Sequence length for prediction')
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='Whether to use image augmentations for training')
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='Whether to use prefetcher for faster data loading')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Whether to drop the last batch in the val data loading')

    # method parameters
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'E3DLSTM', 'e3dlstm', 'MAU', 'mau', 'MIM', 'mim', 
                                'PhyDNet', 'phydnet', 'PredRNN', 'predrnn', 'PredRNNpp',  'predrnnpp', 
                                'PredRNNv2', 'predrnnv2', 'SimVP', 'simvp', 'TAU', 'tau', 'MMVP', 'mmvp', 
                                'SwinLSTM', 'swinlstm', 'swinlstm_d', 'swinlstm_b',
                                'EvolutionConvLSTM', 'evolution_convlstm',
                                'EvolutionTemporalUNet', 'evolutiontemporalunet',
                                'evolution_temporal_unet'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='Path to the default config file')
    parser.add_argument('--model_type', default=None, type=str,
                        help='Name of model for SimVP (default: None)')
    parser.add_argument(
        '--evolution_encoder_checkpoint', default=None, type=str,
        help='ConvLSTM encoder initialization checkpoint; use "none" for '
             'the scratch EvolutionConvLSTM ablation')
    parser.add_argument(
        '--evolution_field_space', default=None, type=str,
        choices=['normalized_dbz', 'linear_z', 'rain_rate'],
        help='Variable space used by the differentiable evolution warp')
    parser.add_argument(
        '--evolution_stop_gradient', action='store_true', default=False,
        help='Detach the evolved field between successive evolution steps')
    parser.add_argument(
        '--evolution_freeze_encoder_epochs', default=None, type=int,
        help='Number of initial epochs that freeze the evolution encoder')
    parser.add_argument(
        '--evolution_encoder_lr', default=None, type=float,
        help='Maximum learning rate for the evolution history encoder')
    parser.add_argument(
        '--evolution_head_lr', default=None, type=float,
        help='Maximum learning rate for the evolution motion head')
    parser.add_argument(
        '--evolution_use_flow_gate', action='store_true', default=False,
        help='Apply a learned spatial confidence gate to raw evolution flow')
    parser.add_argument(
        '--evolution_motion_checkpoint', default=None, type=str,
        help='Initialize evolution encoder and raw motion head from a checkpoint')
    parser.add_argument(
        '--evolution_freeze_motion_epochs', default=None, type=int,
        help='Number of initial epochs that freeze the raw motion head')
    parser.add_argument(
        '--evolution_gate_lr', default=None, type=float,
        help='Maximum learning rate for the evolution flow gate')
    parser.add_argument(
        '--evolution_gate_initial', default=None, type=float,
        help='Initial sigmoid value for a newly created evolution flow gate')
    parser.add_argument(
        '--evolution_gate_supervision_weight', default=None, type=float,
        help='Weight of teacher-forced oracle-scale supervision for the gate')
    parser.add_argument(
        '--evolution_gate_active_threshold', default=None, type=float,
        help='Minimum rain rate included in oracle gate supervision')
    parser.add_argument(
        '--evolution_gate_supervision_only', action='store_true', default=False,
        help='Train only the oracle gate loss for gate pretraining')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate(default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate for SimVP (default: 0.)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to allow overwriting the provided config file with args')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=None, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    # lightning
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)
    parser.add_argument(
        '--motion_eval_only', action='store_true', default=False,
        help='Validate a motion checkpoint with a zero-initialized source head.')
    parser.add_argument(
        '--init_from_ckpt', default=None, type=str,
        help='Initialize model weights from a checkpoint without restoring '
             'optimizer, scheduler, epoch, or global step')

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'fps': False,
        'test': False,
        'skip_test_after_train': False,
        'deterministic': False,
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': './data',
        'dataname': 'mmnist',
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',
        'config_file': None,
        'model_type': 'gSTA',
        'evolution_encoder_checkpoint': None,
        'evolution_freeze_encoder_epochs': None,
        'evolution_encoder_lr': None,
        'evolution_head_lr': None,
        'evolution_use_flow_gate': False,
        'evolution_motion_checkpoint': None,
        'evolution_freeze_motion_epochs': None,
        'evolution_gate_lr': None,
        'evolution_gate_initial': None,
        'evolution_gate_supervision_weight': None,
        'evolution_gate_active_threshold': None,
        'evolution_gate_supervision_only': False,
        'evolution_field_space': None,
        'evolution_stop_gradient': False,
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
        # Lightning parameters
        'gpus': [0],
        'metric_for_bestckpt': 'val_loss',
        'ckpt_path': None,
        'motion_eval_only': False,
        'init_from_ckpt': None,
    }
    return default_values
