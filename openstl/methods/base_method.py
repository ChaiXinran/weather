import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
from openstl.core.precipitation_metrics import PrecipitationEvaluator
from openstl.core.precipitation_loss import PrecipitationR2Loss


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
        self.validation_criterion = nn.MSELoss()
        if args.get('loss_type', 'mse') == 'precipitation_r2':
            self.criterion = PrecipitationR2Loss(
                value_scale=args.get('radar_value_scale', 50.0),
                zr_a=args.get('zr_a', 200.0),
                zr_b=args.get('zr_b', 1.6),
                thresholds=args.get('r2_thresholds', [16.0, 32.0]),
                intensity_weights=args.get(
                    'r2_intensity_weights', [2.0, 3.0]),
                soft_csi_weights=args.get(
                    'r2_soft_csi_weights', [0.005, 0.001]),
                soft_csi_temperature=args.get(
                    'r2_soft_csi_temperature', 0.03),
                huber_beta=args.get('r2_huber_beta', 0.05),
                second_hour_weight=args.get(
                    'r2_second_hour_weight', 1.2),
                soft_csi_mode=args.get('r2_soft_csi_mode', 'micro'),
                segmented_soft_csi_weights=args.get(
                    'r2_segmented_soft_csi_weights', None),
                empty_event_penalty=args.get(
                    'r2_empty_event_penalty', 0.1),
            )
        else:
            self.criterion = nn.MSELoss()
        self.test_outputs = []
        self.precipitation_evaluator = None
        self.test_sample_offset = 0
        self._val_precip = None
        self._val_precip_lead = None

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
        loss = self.validation_criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        if self.hparams.dataname == 'bth_radar':
            self._update_val_precipitation(pred_y, batch_y)
        return loss

    def on_validation_epoch_start(self):
        self._val_precip = None
        self._val_precip_lead = None

    def on_train_epoch_start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 2**20
            self.log('peak_gpu_memory_mb', peak_mb, on_epoch=True)

    @staticmethod
    def _safe_torch_ratio(numerator, denominator):
        return numerator / denominator.clamp_min(1.0)

    def _to_precipitation(self, values):
        values = (values.detach().float()
                  * float(self.hparams.get('radar_value_scale', 50.0)))
        clip_range = self.hparams.get('precip_clip_range', [0.0, 50.0])
        values = values.clamp(float(clip_range[0]), float(clip_range[1]))
        if self.hparams.get('convert_dbz_to_rain', False):
            reflectivity = torch.pow(10.0, values / 10.0)
            values = torch.pow(
                reflectivity / float(self.hparams.get('zr_a', 200.0)),
                1.0 / float(self.hparams.get('zr_b', 1.6)))
        return values

    def _update_val_precipitation(self, pred_y, batch_y):
        pred = self._to_precipitation(pred_y)
        if self.hparams.get('evaluation_truth', 'radar') == 'rain_png':
            # The validation loader returns Radar targets. Rain-PNG replacement
            # is intentionally reserved for the ordered, event-aware test pass.
            true = self._to_precipitation(batch_y)
        else:
            true = self._to_precipitation(batch_y)
        thresholds = self.hparams.get('val_precip_thresholds', [16.0, 32.0])
        period_size = min(10, pred.shape[1])
        periods = [(0, period_size), (period_size, pred.shape[1])]
        state = []
        for start, end in periods:
            period_pred, period_true = pred[:, start:end], true[:, start:end]
            pred_sum = period_pred.sum(dtype=torch.float64)
            true_sum = period_true.sum(dtype=torch.float64)
            values = [pred_sum, true_sum]
            for threshold in thresholds:
                pred_event = period_pred >= float(threshold)
                true_event = period_true >= float(threshold)
                values.extend([
                    (pred_event & true_event).sum(dtype=torch.float64),
                    (pred_event & ~true_event).sum(dtype=torch.float64),
                    (~pred_event & true_event).sum(dtype=torch.float64),
                    pred_event.sum(dtype=torch.float64),
                    true_event.sum(dtype=torch.float64),
                ])
            state.append(torch.stack(values))
        batch_state = torch.stack(state)
        if self._val_precip is None:
            self._val_precip = batch_state
        else:
            self._val_precip += batch_state

        lead_state = []
        for lead in range(pred.shape[1]):
            lead_pred, lead_true = pred[:, lead:lead + 1], true[:, lead:lead + 1]
            values = [
                lead_pred.sum(dtype=torch.float64),
                lead_true.sum(dtype=torch.float64),
            ]
            for threshold in thresholds:
                pred_event = lead_pred >= float(threshold)
                true_event = lead_true >= float(threshold)
                values.extend([
                    (pred_event & true_event).sum(dtype=torch.float64),
                    (pred_event & ~true_event).sum(dtype=torch.float64),
                    (~pred_event & true_event).sum(dtype=torch.float64),
                    pred_event.sum(dtype=torch.float64),
                    true_event.sum(dtype=torch.float64),
                ])
            lead_state.append(torch.stack(values))
        lead_state = torch.stack(lead_state)
        if self._val_precip_lead is None:
            self._val_precip_lead = lead_state
        else:
            self._val_precip_lead += lead_state

    def on_validation_epoch_end(self):
        if self.hparams.dataname != 'bth_radar' or self._val_precip is None:
            return
        state = self._val_precip
        if self.trainer.world_size > 1:
            state = self.all_gather(state).sum(dim=0)
        thresholds = self.hparams.get('val_precip_thresholds', [16.0, 32.0])
        csi_values = {}
        for period_index, period_name in enumerate(('0_1h', '1_2h')):
            values = state[period_index]
            self.log(
                f'val_intensity_ratio_{period_name}',
                self._safe_torch_ratio(values[0], values[1]),
                prog_bar=False)
            offset = 2
            for threshold in thresholds:
                hits, false_alarms, misses, pred_area, true_area = \
                    values[offset:offset + 5]
                label = f'{float(threshold):g}'
                csi = self._safe_torch_ratio(
                    hits, hits + false_alarms + misses)
                csi_values[(period_index, float(threshold))] = csi
                self.log(f'val_csi_{label}_{period_name}', csi)
                self.log(
                    f'val_pod_{label}_{period_name}',
                    self._safe_torch_ratio(hits, hits + misses))
                self.log(
                    f'val_far_{label}_{period_name}',
                    self._safe_torch_ratio(false_alarms, hits + false_alarms))
                self.log(
                    f'val_bias_{label}_{period_name}',
                    self._safe_torch_ratio(
                        hits + false_alarms, hits + misses))
                self.log(
                    f'val_area_ratio_{label}_{period_name}',
                    self._safe_torch_ratio(pred_area, true_area))
                offset += 5
        score = (
            csi_values[(0, 16.0)]
            + csi_values[(0, 32.0)]
            + csi_values[(1, 16.0)]
            + 2.0 * csi_values[(1, 32.0)])
        self.log('val_csi_score', score, prog_bar=True)
        lead_state = self._val_precip_lead
        if self.trainer.world_size > 1:
            lead_state = self.all_gather(lead_state).sum(dim=0)
        for lead_index, values in enumerate(lead_state):
            lead_minutes = (
                (lead_index + 1) * int(self.hparams.get('lead_minutes', 6)))
            self.log(
                f'val_intensity_ratio_lead_{lead_minutes:03d}m',
                self._safe_torch_ratio(values[0], values[1]))
            offset = 2
            for threshold in thresholds:
                hits, false_alarms, misses, pred_area, true_area = \
                    values[offset:offset + 5]
                label = f'{float(threshold):g}'
                lead_csi = self._safe_torch_ratio(
                    hits, hits + false_alarms + misses)
                self.log(
                    f'val_csi_{label}_lead_{lead_minutes:03d}m',
                    lead_csi)
                if lead_minutes in (60, 120) and float(threshold) in (16.0, 32.0):
                    # Short aliases are deliberately progress-bar metrics so
                    # training logs expose the two decision leads every epoch.
                    self.log(
                        f'val_csi{label}_t{lead_minutes}', lead_csi,
                        prog_bar=True)
                self.log(
                    f'val_far_{label}_lead_{lead_minutes:03d}m',
                    self._safe_torch_ratio(
                        false_alarms, hits + false_alarms))
                self.log(
                    f'val_bias_{label}_lead_{lead_minutes:03d}m',
                    self._safe_torch_ratio(
                        pred_area, true_area))
                offset += 5
    
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
