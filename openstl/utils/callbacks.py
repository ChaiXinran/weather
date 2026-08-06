import json
import shutil
import logging
import time
import os.path as osp
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from .main_utils import check_dir, collect_env, print_log, output_namespace


class SetupCallback(Callback):
    def __init__(self, prefix, setup_time, save_dir, ckpt_dir, args, method_info, argv_content=None):
        super().__init__()
        self.prefix = prefix
        self.setup_time = setup_time
        self.save_dir = save_dir
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.config = args.__dict__
        self.argv_content = argv_content
        self.method_info = method_info

    def on_fit_start(self, trainer, pl_module):
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'

        if trainer.global_rank == 0:
            # check dirs
            self.save_dir = check_dir(self.save_dir)
            self.ckpt_dir = check_dir(self.ckpt_dir)
            # setup log
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,
                filename=osp.join(self.save_dir, '{}_{}.log'.format(self.prefix, self.setup_time)),
                filemode='a', format='%(asctime)s - %(message)s')
            # print env info
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
            sv_param = osp.join(self.save_dir, 'model_param.json')
            with open(sv_param, 'w') as file_obj:
                json.dump(self.config, file_obj)

            print_log(output_namespace(self.args))
            if self.method_info is not None:
                info, flops, fps, dash_line = self.method_info
                print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)


class EpochEndCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        self.avg_train_loss = trainer.callback_metrics.get('train_loss')

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        lr = trainer.optimizers[0].param_groups[0]['lr']
        avg_val_loss = trainer.callback_metrics.get('val_loss')

        if hasattr(self, 'avg_train_loss'):
            elapsed = time.perf_counter() - self.epoch_start_time
            peak_mb = trainer.callback_metrics.get('peak_gpu_memory_mb')
            print_log(
                f"Epoch {trainer.current_epoch}: Lr: {lr:.7f} | "
                f"Train Loss: {self.avg_train_loss:.7f} | "
                f"Vali Loss: {avg_val_loss:.7f} | "
                f"Wall: {elapsed:.1f}s | Peak GPU: {peak_mb:.1f} MiB")

class BestCheckpointCallback(ModelCheckpoint):
    def __init__(self, *args, alias_name='best.ckpt', **kwargs):
        self.alias_name = alias_name
        super().__init__(*args, **kwargs)

    @property
    def state_key(self):
        # Lightning requires distinct state keys when several checkpoint
        # monitors are registered on the same trainer.
        monitor = self.monitor or 'none'
        return f'{self.__class__.__qualname__}[{monitor}][{self.alias_name}]'

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.best_model_path and trainer.global_rank == 0:
            shutil.copy(
                self.best_model_path,
                osp.join(osp.dirname(self.best_model_path), self.alias_name))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        if self.best_model_path and trainer.global_rank == 0:
            shutil.copy(
                self.best_model_path,
                osp.join(osp.dirname(self.best_model_path), self.alias_name))


class CheckpointAliasCallback(Callback):
    """Copy stable checkpoint aliases after all ModelCheckpoint hooks run."""

    @staticmethod
    def _copy_aliases(trainer):
        if not trainer.is_global_zero:
            return
        for callback in trainer.checkpoint_callbacks:
            best_path = getattr(callback, 'best_model_path', '')
            alias_name = getattr(callback, 'alias_name', '')
            if best_path and alias_name:
                shutil.copy(
                    best_path, osp.join(osp.dirname(best_path), alias_name))

    def on_train_epoch_end(self, trainer, pl_module):
        self._copy_aliases(trainer)

    def on_validation_end(self, trainer, pl_module):
        self._copy_aliases(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._copy_aliases(trainer)
