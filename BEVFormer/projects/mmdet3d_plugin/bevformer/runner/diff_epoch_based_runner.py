# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from mmcv.runner.builder import RUNNERS
from mmcv.runner.epoch_based_runner import EpochBasedRunner

@RUNNERS.register_module()
class DiffEpochBasedRunner(EpochBasedRunner):
    def __init__(self,
                 model,
                 model_target=None,
                 bev_diffuser=None,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):            
        super().__init__(model,
                        batch_processor,
                        optimizer,
                        work_dir,
                        logger,
                        meta,
                        max_iters,
                        max_epochs)
        self.model_target = model_target
        if self.model_target:
            self.model_target.eval()
            
        self.bev_diffuser = bev_diffuser
        if self.bev_diffuser:
            self.bev_diffuser.eval()
        
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, model_target=self.model_target, bev_diffuser=self.bev_diffuser, progress=1.0 * (self.epoch+1) / self.max_epochs, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        
    
    def train(self, data_loader, **kwargs):
        if self.bev_diffuser and self.bev_diffuser.auto_denoise_timesteps:
            max_step, min_step = 1000, 0
            self.bev_diffuser.denoise_timesteps = int(max_step - (max_step - min_step) * (1.0 * self.epoch / self.max_epochs))
            self.bev_diffuser.num_inference_steps = min(self.bev_diffuser.num_inference_steps, self.bev_diffuser.denoise_timesteps)
        super().train(data_loader, **kwargs)