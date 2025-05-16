# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())
            
@HOOKS.register_module()
class UpdateTarget(Hook):
    
    def __init__(self, iter_interval=0, epoch_interval=0):
        self.iter_interval=iter_interval
        self.epoch_interval=epoch_interval

    def _update_target(self, runner):
        state_dict = runner.model.state_dict()
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        runner.model_target.load_state_dict(state_dict)
    
    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.iter_interval):
            self._update_target(runner)
            
    def after_train_epoch(self, runner):
        if self.every_n_epochs(runner, self.epoch_interval):
            self._update_target(runner)
            
    # def before_run(self, runner):
    #     self._update_target(runner)