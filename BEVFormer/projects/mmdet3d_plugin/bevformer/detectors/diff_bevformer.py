# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
import time
import copy
import numpy as np
import mmdet3d
from .bevformer import BEVFormer

from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class DiffBEVFormer(BEVFormer): 
    
    def train_step(self, data, optimizer, model_target=None, bev_diffuser=None, progress=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data, model_target=model_target, bev_diffuser=bev_diffuser)
        
        weight = 0.5
        # if progress is not None:
        #     weight = max(1 - progress*2, 0.1)
        loss, log_vars = self._parse_losses_mix(losses, weight=weight)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      given_bev=None,
                      model_target=None,
                      bev_diffuser=None,
                      **kwargs,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bev_target = None
        if bev_diffuser:
            assert model_target is not None 
            bev_target = model_target(return_loss=False, only_bev=True, img=img, img_metas=img_metas).detach()
            
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev, given_bev, model_target, bev_diffuser, bev_target, **kwargs)

        losses.update(losses_pts)
        return losses
            
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None,
                          given_bev=None,
                          model_target=None,
                          bev_diffuser=None,
                          bev_target=None,
                          **kwargs):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        bev = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev, only_bev=True
        )
        
        losses = dict()
        
        outs = self.pts_bbox_head(
        pts_feats, img_metas, prev_bev, given_bev=bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        
        if bev_diffuser:
            assert bev_target is not None   
              
            def get_classifier_gradient(x):
                x_in = x.detach().requires_grad_(True)
                x_in = x_in.permute(0, 2, 3, 1).reshape(-1, self.pts_bbox_head.bev_h * self.pts_bbox_head.bev_w, bev.shape[-1])
                outs = model_target.pts_bbox_head(pts_feats, img_metas, prev_bev=prev_bev, given_bev=x_in)
                losses = model_target.pts_bbox_head.loss(
                    gt_bboxes_list=gt_bboxes_3d,
                    gt_labels_list=gt_labels_3d,
                    preds_dicts=outs,
                    img_metas=img_metas
                )
                loss, _ = self._parse_losses(losses)
                gradient = torch.autograd.grad(loss, x_in)[0]
                gradient = gradient.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
                return gradient
            
            def get_condition():
                cond = {}
                if 'layout_obj_classes' in kwargs:
                    cond['obj_class'] = torch.stack(kwargs['layout_obj_classes'])
                if 'layout_obj_bboxes' in kwargs:
                    cond['obj_bbox'] = torch.stack(kwargs['layout_obj_bboxes'])
                if 'layout_obj_is_valid' in kwargs:
                    cond['is_valid_obj'] = torch.stack(kwargs['layout_obj_is_valid']) 
                if 'layout_obj_names' in kwargs:
                    cond['obj_name'] = torch.stack(kwargs['layout_obj_names'])
                if 'default_obj_names' in kwargs:
                    cond['default_obj_names'] = torch.stack(kwargs['default_obj_names'])           
                return cond

            bev_ = bev_target.detach()
            bev_ = bev_.reshape(-1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, bev.shape[-1]).permute(0, 3, 1, 2)
            bev_ = bev_diffuser(bev_, get_condition(), grad_fn=get_classifier_gradient)
            bev_ = bev_.permute(0, 2, 3, 1).reshape(-1, self.pts_bbox_head.bev_h*self.pts_bbox_head.bev_w, bev.shape[-1])
            loss_bev = F.mse_loss(bev.float(), bev_.detach().float(), reduction="mean")
            
            losses['loss_bev'] = loss_bev*100
    
        return losses
    
    def _parse_losses_mix(self, losses, weight=0.5):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        total_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        bev_loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key and 'bev' in _key)
        task_loss = total_loss - bev_loss
        
        loss = (1-weight) * task_loss + weight * bev_loss

        log_vars['loss'] = loss
        log_vars['task_loss'] = task_loss
        # log_vars['gq_loss'] = gq_loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars