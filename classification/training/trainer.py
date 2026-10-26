"""Controlled one-model, one-size, one-fold training for CPU checks and HPC."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .grouped_data import CLASS_NAMES, make_fold_loaders, fingerprint
from .operational_metrics import select_threshold, report


@dataclass
class ControlledConfig:
    model: str = 'effnet_b2'
    patch_size: int = 256
    fold: int = 1
    seed: int = 42
    dataset_root: str = 'stage2_results/training_patches'
    folds_dir: str = 'stage2_results/grouped_folds_v1'
    output_root: str = 'stage2_results/controlled_runs'
    device: str = 'cuda'
    epochs: int = 50
    adaptation_epochs: int = 5
    batch_size: int = 16
    accumulation_steps: int = 4
    num_workers: int = 4
    cache_ram: bool = False
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    stem_lr: float = 1e-4
    head_adaptation_lr: float = 1e-3
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    recall_min: float = .90
    recall_target: float = .95
    speckle_std: float = .05
    intensity_min: float = .9
    intensity_max: float = 1.1
    smoke: bool = False
    max_train_batches: int = 0
    max_val_batches: int = 0

    def validate(self):
        if self.patch_size not in (256, 384, 512) or self.fold not in range(1, 6):
            raise ValueError('Expected native patch size and fold 1..5')
        if min(self.epochs, self.adaptation_epochs, self.batch_size, self.accumulation_steps) < 1:
            raise ValueError('Epoch and batch counts must be positive')
        if self.adaptation_epochs >= self.epochs or self.num_workers < 0:
            raise ValueError('Invalid adaptation duration or workers')
        if not 0 < self.recall_min <= self.recall_target <= 1:
            raise ValueError('Invalid recall policy')
        if min(self.max_train_batches, self.max_val_batches, self.speckle_std) < 0:
            raise ValueError('Negative limit or noise scale')
        if not 0 < self.intensity_min <= self.intensity_max:
            raise ValueError('Invalid augmentation interval')
        for value in (self.weight_decay, self.gradient_clip, self.stem_lr,
                      self.head_adaptation_lr, self.backbone_lr, self.head_lr):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Optimization settings must be finite and positive')
        if not self.smoke and (self.epochs != 50 or self.adaptation_epochs != 5
                               or self.max_train_batches or self.max_val_batches
                               or self.batch_size*self.accumulation_steps != 64):
            raise ValueError('Controlled profile requires 50 epochs, 5 adaptation epochs, effective batch 64 and full data; use --smoke for engineering checks')


class SARAugmentation:
    """Intensity perturbation defaults, not a physical SAR radiometry model."""
    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        if torch.rand(()) < .5:
            x = x.flip(-1)
        if torch.rand(()) < .5:
            x = x.flip(-2)
        if torch.rand(()) < .5:
            x = torch.rot90(x, int(torch.randint(1, 4, ()).item()), (-2, -1))
        if torch.rand(()) < .3:
            x = x * (1 + torch.randn_like(x)*self.config.speckle_std)
        if torch.rand(()) < .3:
            x = x * float(torch.empty(()).uniform_(self.config.intensity_min, self.config.intensity_max))
        return x.clamp(0, 1).contiguous()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(value, path):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def save_checkpoint(value, path):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


class Trainer:
    """Epoch-boundary restart with strict data/code/configuration identity."""
    def __init__(self, config, resume=None, model=None, loaders=None):
        if not isinstance(config, ControlledConfig):
            raise TypeError('Use ControlledConfig with the controlled CLI')
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; fix HPC environment or use --device cpu --smoke')
        seed_all(config.seed)
        if (model is not None or loaders is not None) and not config.smoke:
            raise ValueError('Dependency injection is restricted to engineering tests')
        self.loaders = loaders or make_fold_loaders(
            Path(config.dataset_root)/f'ps{config.patch_size}_overlap50', config.folds_dir,
            config.fold, config.batch_size, config.num_workers, config.seed, SARAugmentation(config),
            cache_ram=config.cache_ram)
        injected = model is not None
        if model is None:
            from .models import build_model
            model = build_model(config.model)
        self.model = model.to(self.device)
        self.head = (model.head_module if hasattr(model, 'head_module') else model.fc if hasattr(model, 'fc')
                     else model.head if hasattr(model, 'head') else model.classifier)
        self.stem = model.conv1 if hasattr(model, 'conv1') else model.features[0]
        selected = {id(p) for m in (self.stem, self.head) for p in m.parameters()}
        self.groups = [list(self.stem.parameters()), list(self.head.parameters()),
                       [p for p in model.parameters() if id(p) not in selected]]
        self.optimizer = torch.optim.AdamW([dict(params=g, lr=0.) for g in self.groups], weight_decay=config.weight_decay)
        # torch.amp.GradScaler is unavailable in the HPC environment's PyTorch 2.2.
        scaler_type = getattr(torch.amp, 'GradScaler', None)
        self.scaler = (scaler_type('cuda', enabled=self.device.type == 'cuda')
                       if scaler_type is not None
                       else torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda'))
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.loaders['train'].dataset.positive_weights().to(self.device))
        self.history, self.best_key, self.best_epoch, self.best_ap = [], None, None, -1.
        self.best_weights, self.best_ap_weights, self.best_ap_epoch = None, None, None
        self.steps, self.amp_skipped_steps, self.elapsed, self.start_epoch = 0, 0, 0., 1
        self.scheduler = dict(policy='linear adaptation warmup then epoch cosine to 1% LR', epoch=0)
        self.stop_requested = False
        name = f'{config.model}_p{config.patch_size}_fold{config.fold:02d}_seed{config.seed}'
        self.output = Path(config.output_root)/(('smoke_' if config.smoke else '')+name)
        if resume is None and self.output.exists():
            raise FileExistsError(f'Run exists: {self.output}; use --resume')
        if resume is not None and not self.output.exists():
            raise FileNotFoundError('Resume requires the original run directory')
        self.output.mkdir(parents=True, exist_ok=True)
        self.contract = self.make_contract(injected)
        if resume is not None:
            self.restore(self.output/'last.pt' if resume == 'auto' else Path(resume))
        else:
            save_json(self.contract, self.output/'config.json')
            save_checkpoint(self.state(0), self.output/'last.pt')

    def make_contract(self, injected):
        import torchvision
        source_hashes = {name: fingerprint(Path(__file__).with_name(name)) for name in
                         ('trainer.py', 'grouped_data.py', 'models.py', 'operational_metrics.py')}
        manifest = Path(self.config.folds_dir)/'scene_folds.csv'
        if manifest.exists():
            source_hashes['scene_folds.csv'] = fingerprint(manifest)
            source_hashes['fold_audit.json'] = fingerprint(manifest.with_name('audit.json'))
        weights = dict(effnet_b0='EfficientNet_B0_Weights.IMAGENET1K_V1',
                       effnet_b2='EfficientNet_B2_Weights.IMAGENET1K_V1',
                       convnext_tiny='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
                       resnet50='ResNet50_Weights.IMAGENET1K_V2',
                       mobilenet_v3_large='MobileNet_V3_Large_Weights.IMAGENET1K_V2',
                       swin_tiny='Swin_T_Weights.IMAGENET1K_V1',
                       effnet_b2_cbam='EfficientNet_B2_Weights.IMAGENET1K_V1 (+ random CBAM)',
                       effnet_b2_cbam_transformer='EfficientNet_B2_Weights.IMAGENET1K_V1 (+ random CBAM, Transformer)')
        return dict(config=asdict(self.config), source_hashes=source_hashes,
                    class_names=list(CLASS_NAMES), pos_weight=self.loss.pos_weight.cpu().tolist(),
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    train_class_counts=self.loaders['train'].dataset.targets.sum(0).tolist(),
                    initialization='injected engineering model' if injected else weights[self.config.model],
                    torch_version=str(torch.__version__), torchvision_version=torchvision.__version__,
                    numpy_version=np.__version__, cuda_version=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
                    augmentation=dict(hflip_p=.5, vflip_p=.5, rot90_p=.5, speckle_p=.3, intensity_p=.3,
                                      ranges='See config; engineering defaults pending HPC validation'),
                    selection='Recall constraint then F2, AP, Precision, lower FN, lower loss; exact ties retain earliest epoch',
                    pr_auc='Non-interpolated average precision (AP)', non_oil_threshold=.5,
                    calibration='none; raw probabilities',
                    started_at=datetime.now(timezone.utc).isoformat())

    def set_phase(self, epoch):
        c = self.config
        adaptation = epoch <= c.adaptation_epochs
        for p in self.model.parameters():
            p.requires_grad_(not adaptation)
        for group in self.groups[:2]:
            for p in group:
                p.requires_grad_(True)
        if adaptation:
            scale = epoch/c.adaptation_epochs
            rates = [c.stem_lr*scale, c.head_adaptation_lr*scale, 0.]
        else:
            progress = (epoch-c.adaptation_epochs-1)/max(c.epochs-c.adaptation_epochs-1, 1)
            scale = .01 + .99*(1+math.cos(math.pi*progress))/2
            rates = [c.backbone_lr*scale, c.head_lr*scale, c.backbone_lr*scale]
        for group, rate in zip(self.optimizer.param_groups, rates):
            group['lr'] = rate
        self.scheduler['epoch'] = epoch
        self.model.train()
        if adaptation:
            self.model.eval()  # Freeze backbone BatchNorm stats and stochastic layers.
            self.stem.train()
            self.head.train()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def train_epoch(self):
        loader, c = self.loaders['train'], self.config
        limit = min(len(loader), c.max_train_batches or len(loader))
        total_loss, samples = 0., 0
        completed_before = self.steps
        available_samples = min(len(loader.dataset), limit*c.batch_size)
        effective_batch = c.batch_size*c.accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(loader):
            if step >= limit:
                break
            x, y = x.to(self.device), y.to(self.device)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite training data')
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                logits = self.model(x)
                loss = self.loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite logits or loss')
            group_start = (step//c.accumulation_steps)*effective_batch
            group_size = min(effective_batch, available_samples-group_start)
            # Normalize before AMP scaling so a full batch does not multiply
            # the loss scale by the effective batch size during backward.
            self.scaler.scale(loss*(len(x)/group_size)).backward()
            samples += len(x)
            total_loss += float(loss.detach())*len(x)
            if (step+1) % c.accumulation_steps == 0 or step+1 == limit:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), c.gradient_clip,
                                         error_if_nonfinite=False)
                # GradScaler detects overflow in step() and skips the update.
                # Gradient clipping must not abort before it can do so.
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() < scale_before:
                    self.amp_skipped_steps += 1
                else:
                    self.steps += 1
                self.optimizer.zero_grad(set_to_none=True)
        if self.steps == completed_before:
            raise RuntimeError('No optimizer updates completed; inspect AMP overflow and input data')
        return total_loss/samples, samples

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        logits_list, targets_list, loss_sum = [], [], 0.
        for step, (x, y) in enumerate(self.loaders['validation']):
            if self.config.max_val_batches and step >= self.config.max_val_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('Nonfinite validation data')
            with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                logits = self.model(x)
                loss = self.loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError('Nonfinite validation output')
            logits_list.append(logits.float().cpu())
            targets_list.append(y.cpu())
            loss_sum += float(loss)*len(x)
        logits, targets = torch.cat(logits_list), torch.cat(targets_list)
        probs = logits.sigmoid().numpy()
        selected = select_threshold(targets[:, 0].numpy(), probs[:, 0], self.config.recall_min)
        result = report(targets.numpy(), probs, selected['threshold'])
        result.update(selected=selected, desired_target=select_threshold(targets[:, 0].numpy(), probs[:, 0], self.config.recall_target),
                      reference_05=report(targets.numpy(), probs, .5), validation_loss=loss_sum/len(targets))
        return result, logits.numpy(), targets.numpy(), probs

    def state(self, epoch):
        return dict(epoch=epoch, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                    scaler=self.scaler.state_dict(), scheduler=self.scheduler, contract=self.contract,
                    history=self.history, best_key=self.best_key, best_epoch=self.best_epoch, best_ap=self.best_ap,
                    best_weights=self.best_weights, best_ap_weights=self.best_ap_weights, best_ap_epoch=self.best_ap_epoch,
                    steps=self.steps, amp_skipped_steps=self.amp_skipped_steps,
                    elapsed=self.elapsed, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None,
                    loader_rng={k: v.generator.get_state() for k, v in self.loaders.items()})

    def restore(self, path):
        # Only load trusted project checkpoints: weights_only=False includes RNG Python objects.
        if path.resolve() != (self.output/'last.pt').resolve():
            raise ValueError('Resume uses the original last.pt; best checkpoints are inference artifacts')
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key in ('config', 'source_hashes', 'pos_weight', 'torch_version', 'torchvision_version', 'numpy_version', 'initialization'):
            if state['contract'][key] != self.contract[key]:
                raise ValueError(f'Resume contract mismatch: {key}')
        self.contract = state['contract']
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        self.scheduler = state['scheduler']
        self.history, self.best_key, self.best_epoch = state['history'], state['best_key'], state['best_epoch']
        self.steps, self.elapsed, self.best_ap = state['steps'], state['elapsed'], state['best_ap']
        self.amp_skipped_steps = state['amp_skipped_steps']
        self.best_weights, self.best_ap_weights = state['best_weights'], state['best_ap_weights']
        self.best_ap_epoch = state['best_ap_epoch']
        self.start_epoch = state['epoch']+1
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        for key, loader in self.loaders.items():
            loader.generator.set_state(state['loader_rng'][key])

    def export_epoch(self, epoch, result, logits, targets, probs):
        frame = self.loaders['validation'].dataset.df.iloc[:len(targets)][['patch_id', 'scene_id']].copy()
        frame['fold'], frame['patch_size'], frame['epoch'] = self.config.fold, self.config.patch_size, epoch
        frame['oil_threshold'] = result['selected']['threshold']
        for i, name in enumerate(CLASS_NAMES):
            frame[f'true_{name}'], frame[f'logit_{name}'], frame[f'prob_{name}'] = targets[:, i].astype(int), logits[:, i], probs[:, i]
            frame[f'pred_{name}'] = (probs[:, i] >= (result['selected']['threshold'] if i == 0 else .5)).astype(int)
        path = self.output/f'validation_epoch_{epoch:03d}.parquet'
        tmp = path.with_suffix('.parquet.tmp')
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def summarize(self):
        complete = len(self.history) == self.config.epochs
        save_json(self.history, self.output/'history.json')
        save_json(dict(status='complete' if complete else 'paused', smoke=self.config.smoke,
                       best_epoch=self.best_epoch, completed_epochs=len(self.history),
                       best=self.history[self.best_epoch-1] if self.best_epoch else None,
                       epoch_seconds=self.elapsed, epoch_gpu_hours=self.elapsed/3600 if self.device.type == 'cuda' else None,
                       timing_scope='Train, validation, prediction export; excludes setup, checkpoint I/O and lost partial epochs',
                       updated_at=datetime.now(timezone.utc).isoformat()), self.output/'metrics.json')
        rows = [dict(epoch=r['epoch'], train_loss=r['train_loss'], validation_loss=r['validation_loss'],
                     oil_recall=r['selected']['recall'], oil_f2=r['selected']['f2'], oil_ap=r['selected']['ap'],
                     oil_precision=r['selected']['precision'], oil_fn=r['oil']['fn'],
                     routed_fraction=r['routed_fraction'], epoch_seconds=r['epoch_seconds'], optimizer_steps=r['optimizer_steps']) for r in self.history]
        pd.DataFrame(rows).to_csv(self.output/'history.csv', index=False)
        if self.best_epoch:
            # Regenerate inference checkpoints from committed last.pt state after interruption.
            for filename, weights, epoch in [('best_operational.pt', self.best_weights, self.best_epoch),
                                              ('best_pr_auc.pt', self.best_ap_weights, self.best_ap_epoch)]:
                save_checkpoint(dict(model=weights, epoch=epoch, contract=self.contract,
                                     metrics=self.history[epoch-1], purpose='inference; resume from last.pt'),
                                self.output/filename)
            frame = pd.read_parquet(self.output/f'validation_epoch_{self.best_epoch:03d}.parquet')
            frame.to_parquet(self.output/'val_predictions.parquet', index=False)
            frame[(frame.true_oil == 1) & (frame.pred_oil == 0)].to_parquet(self.output/'oil_false_negatives.parquet', index=False)
            frame[(frame.true_oil == 0) & (frame.pred_oil == 1)].to_parquet(self.output/'oil_false_positives.parquet', index=False)

    def fit(self, stop_after=None):
        """Pause at an epoch boundary on SIGUSR1; timeout otherwise replays partial epoch."""
        previous = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, lambda *_: setattr(self, 'stop_requested', True))
        try:
            for epoch in range(self.start_epoch, self.config.epochs+1):
                self.sync()
                started = time.perf_counter()
                if self.device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.set_phase(epoch)
                train_loss, samples = self.train_epoch()
                self.sync()
                train_seconds = time.perf_counter()-started
                val_started = time.perf_counter()
                result, logits, targets, probs = self.validate()
                self.sync()
                val_seconds = time.perf_counter()-val_started
                selected = result['selected']
                key = (selected['f2'], selected['ap'], selected['precision'], -result['oil']['fn'], -result['validation_loss'])
                improved = self.best_key is None or key > self.best_key
                ap_improved = selected['ap'] > self.best_ap
                if improved:
                    self.best_key, self.best_epoch = key, epoch
                    self.best_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if ap_improved:
                    self.best_ap = selected['ap']
                    self.best_ap_epoch = epoch
                    self.best_ap_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.export_epoch(epoch, result, logits, targets, probs)
                seconds = time.perf_counter()-started
                self.elapsed += seconds
                result.update(epoch=epoch, train_loss=train_loss, train_samples=samples,
                              train_seconds=train_seconds, validation_seconds=val_seconds, epoch_seconds=seconds,
                              validation_pipeline_patches_per_second=len(targets)/val_seconds,
                              optimizer_steps=self.steps, learning_rates=[g['lr'] for g in self.optimizer.param_groups],
                              amp_skipped_steps=self.amp_skipped_steps,
                              trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None)
                self.history.append(result)
                state = self.state(epoch)
                save_checkpoint(state, self.output/'last.pt')
                self.summarize()
                print(json.dumps(dict(epoch=epoch, oil_recall=selected['recall'], oil_f2=selected['f2'], threshold=selected['threshold'])), flush=True)
                if self.stop_requested or (stop_after is not None and epoch >= stop_after):
                    break
            self.summarize()
        except Exception as error:
            save_json(dict(error=type(error).__name__, message=str(error), time=datetime.now(timezone.utc).isoformat()), self.output/'failure.json')
            raise
        finally:
            signal.signal(signal.SIGUSR1, previous)
