import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
from openstl.core.precipitation_metrics import PrecipitationEvaluator


class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []
        self.precipitation_evaluator = None
        self.test_sample_offset = 0

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        if self.hparams.dataname == 'bth_radar':
            if self.precipitation_evaluator is None:
                self.precipitation_evaluator = PrecipitationEvaluator(
                    lead_count=self.hparams.aft_seq_length,
                    thresholds=self.hparams.get(
                        'precip_thresholds', [20, 30, 35, 40, 45]),
                    value_scale=self.hparams.get('radar_value_scale', 50.0),
                    value_unit=self.hparams.get('precip_value_unit', 'dBZ'),
                    lead_minutes=self.hparams.get('lead_minutes', 6),
                    clip_range=self.hparams.get(
                        'precip_clip_range', [0.0, 50.0]),
                    case_threshold=self.hparams.get(
                        'case_threshold', 35.0),
                    case_count=self.hparams.get('case_count', 3),
                    convert_dbz_to_rain=self.hparams.get(
                        'convert_dbz_to_rain', False),
                    zr_a=self.hparams.get('zr_a', 200.0),
                    zr_b=self.hparams.get('zr_b', 1.6),
                    wet_threshold=self.hparams.get('wet_threshold', 0.1),
                    grid_spacing_km=self.hparams.get(
                        'grid_spacing_km', 10.0),
                    neighborhood_windows=self.hparams.get(
                        'neighborhood_windows', [1, 3, 5]),
                    object_iou_threshold=self.hparams.get(
                        'object_iou_threshold', 0.1),
                    bootstrap_repetitions=self.hparams.get(
                        'bootstrap_repetitions', 2000),
                    bootstrap_seed=self.hparams.get(
                        'bootstrap_seed', 42),
                    true_is_rain=self.hparams.get(
                        'evaluation_truth', 'radar') == 'rain_png',
                    event_id_source=getattr(
                        self.trainer.datamodule.test_loader.dataset,
                        'event_id_source', 'unassigned'),
                )
            batch_size = batch_x.shape[0]
            dataset = self.trainer.datamodule.test_loader.dataset
            sample_indices = list(range(
                self.test_sample_offset,
                self.test_sample_offset + batch_size))
            evaluation_true = batch_y.detach().cpu().numpy()
            if self.hparams.get('evaluation_truth', 'radar') == 'rain_png':
                evaluation_true = dataset.rain_targets(sample_indices)
            event_ids = [
                dataset.event_id_for_sample(index)
                for index in sample_indices
            ]
            self.precipitation_evaluator.update(
                pred_y.detach().cpu().numpy(),
                evaluation_true,
                batch_x.detach().cpu().numpy(),
                event_ids=event_ids,
                sample_ids=sample_indices,
            )
            self.test_sample_offset += batch_size
            return {'batch_size': batch_size}
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_start(self):
        self.test_outputs.clear()
        self.precipitation_evaluator = None
        self.test_sample_offset = 0
        if (self.hparams.dataname == 'bth_radar'
                and self.trainer.world_size != 1):
            raise RuntimeError(
                'BTH Radar event-level evaluation currently requires a '
                'single test process to preserve exact sample/event ordering.')

    def on_test_epoch_end(self):
        if self.hparams.dataname == 'bth_radar':
            if self.precipitation_evaluator is None:
                return {}
            folder_path = check_dir(osp.join(
                self.hparams.save_dir, 'saved', 'precipitation_evaluation'))
            report = self.precipitation_evaluator.save(folder_path)
            overall = report['model']['overall']
            print_log(
                'Radar evaluation: '
                f'MAE={overall["mae"]:.4f} {self.hparams.get("precip_value_unit", "dBZ")}, '
                f'RMSE={overall["rmse"]:.4f} {self.hparams.get("precip_value_unit", "dBZ")}')
            for threshold, values in overall['thresholds'].items():
                print_log(
                    f'Threshold {threshold}: CSI={values["csi"]}, '
                    f'POD={values["pod"]}, FAR={values["far"]}, '
                    f'Bias={values["bias"]}')
            self.precipitation_evaluator = None
            self.test_sample_offset = 0
            return report
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        self.test_outputs.clear()
        return results_all
