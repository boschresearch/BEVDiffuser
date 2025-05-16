# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from diffusers
#   (https://github.com/huggingface/diffusers)
# Copyright (c) 2022 diffusers authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

'''
Following code is adapted from 
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
'''

import argparse
import os, sys
import time

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import custom_encode_mask_results, collect_results_cpu
from mmdet.apis import set_random_seed

from scheduler_utils import DDIMGuidedScheduler
from model_utils import get_bev_model, build_unet
from layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel

logger = get_logger(__name__, log_level="INFO")

def parse_args():
     # put all arg parse here
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument('--bev_config', 
                        default="",
                        help='test config file path')
    
    parser.add_argument('--bev_checkpoint', 
                        default="",
                        help='checkpoint file')
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        choices=[
            "CompVis/stable-diffusion-v1-4",
            "stabilityai/stable-diffusion-2-1"
        ],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="The checkpoint directory of unet.",
    )


    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'sample' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    
    parser.add_argument(
        "--use_classifier_guidence",
        action='store_true',
        help="whether to use classifier guidence",
    )
    
    parser.add_argument(
        '--noise_timesteps', 
        type=int, 
        default=0, 
        help='The number of timesteps to add noise.')
    
    parser.add_argument(
        '--denoise_timesteps', 
        type=int, 
        default=5, 
        help='The number of timesteps to denoise.')
    
    parser.add_argument(
        '--num_inference_steps', 
        type=int, 
        default=5, 
        help='The number of diffusion steps to run the unet.')
    
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

    
def test():
    args = parse_args()

    bev_cfg = Config.fromfile(args.bev_config)
    
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=False)
        
    if args.launcher != 'none':
        init_dist(args.launcher, **bev_cfg.dist_params)
        
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMGuidedScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    if args.prediction_type is not None:
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    
    bev_model = get_bev_model(args)
    if not args.use_classifier_guidence:
        bev_model.requires_grad_(False)
    
    unet = build_unet(bev_cfg.unet)
    unet.from_pretrained(args.checkpoint_dir, subfolder="unet")
    unet.to(bev_model.device, dtype=torch.float32)
    unet.requires_grad_(False) 
    unet.eval()
    
    bev_cfg.data.test.test_mode = True
    bev_cfg.data.test.load_annos = True
    dataset = build_dataset(bev_cfg.data.test,
                            default_args={
                                        'pc_range': bev_cfg.point_cloud_range,
                                        'use_3d_bbox': bev_cfg.use_3d_bbox,
                                        'num_classes': bev_cfg.num_classes,
                                        'num_bboxes': bev_cfg.num_bboxes,
                                    })
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=bev_cfg.data.samples_per_gpu,
        workers_per_gpu=bev_cfg.data.workers_per_gpu,
        dist=(args.launcher != 'none'),
        shuffle=False,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
  
    save_path = os.path.join('../../test', args.bev_config.split('/')[-1].split('.')[-2], args.checkpoint_dir.split('/')[-2], args.checkpoint_dir.split('/')[-1])
        
    evaluate(unet=unet,
             bev_model=bev_model,
             noise_scheduler=noise_scheduler,
             dataset=dataset,
             dataloader=dataloader,
             bev_cfg=bev_cfg,
             eval=args.eval,
             save_path=save_path,
             noise_timesteps=args.noise_timesteps,
             denoise_timesteps=args.denoise_timesteps,
             num_inference_steps=args.num_inference_steps,
             use_classifier_guidence=args.use_classifier_guidence)


def evaluate(unet,
             bev_model,
             noise_scheduler,
             dataset,
             dataloader,
             bev_cfg,
             eval='bbox',
             save_path='',
             noise_timesteps=0,
             denoise_timesteps=0,
             num_inference_steps=0,
             use_classifier_guidence=False):
    
    def get_classifier_gradient(x, **kwargs):
        x_ = x.detach().requires_grad_(True)
        x_ = x_.permute(0, 2, 3, 1)
        x_ = x_.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        loss = bev_model(return_loss=False, only_bev=False, given_bev=x_, return_eval_loss=True, **kwargs)
        gradient = torch.autograd.grad(loss, x_)[0]
        gradient = gradient.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        gradient = gradient.permute(0, 3, 1, 2)
        return gradient
    
    def get_condition(batch, use_cond=True):
        cond = {}
        if 'layout_obj_classes' in batch:
            cond['obj_class'] = torch.stack(batch['layout_obj_classes'].data[0])
        if 'layout_obj_bboxes' in batch:
            cond['obj_bbox'] = torch.stack(batch['layout_obj_bboxes'].data[0])
        if 'layout_obj_is_valid' in batch:
            cond['is_valid_obj'] = torch.stack(batch['layout_obj_is_valid'].data[0]) 
        if 'layout_obj_names' in batch:
            cond['obj_name'] = torch.stack(batch['layout_obj_names'].data[0])
        
        if not use_cond:
            if isinstance(unet, LayoutDiffusionUNetModel):
                if 'obj_class' in unet.layout_encoder.used_condition_types:
                    cond['obj_class'] = torch.ones_like(cond['obj_class']).fill_(unet.layout_encoder.num_classes_for_layout_object - 1)
                    cond['obj_class'][:, 0] = unet.layout_encoder.num_classes_for_layout_object - 2
                if 'obj_name' in unet.layout_encoder.used_condition_types:
                    cond['obj_name'] = torch.stack(batch['default_obj_names'].data[0])
                if 'obj_bbox' in unet.layout_encoder.used_condition_types:
                    cond['obj_bbox'] = torch.zeros_like(cond['obj_bbox'])
                    if unet.layout_encoder.use_3d_bbox:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
                    else:
                        cond['obj_bbox'][:, 0] = torch.FloatTensor([0, 0, 1, 1])
                cond['is_valid_obj'] = torch.zeros_like(cond['is_valid_obj'])
                cond['is_valid_obj'][:, 0] = 1.0 
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond[key] = value.to(latents.device)            
        return cond
    
    det_res_path = f"{noise_timesteps}_{denoise_timesteps}_{num_inference_steps}"
    bbox_results = []
    mask_results = []
    have_mask = False
    
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for step, batch in enumerate(dataloader):
        
        latents = bev_model(return_loss=False, only_bev=True, **batch).detach()
        
        latents = latents.reshape(-1, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_)
        
        latents = latents.permute(0, 3, 1, 2)
        
        if noise_timesteps > 0:
            if noise_timesteps > 1000:
                latents = torch.randn_like(latents)
                latents = latents * noise_scheduler.init_noise_sigma
            else:   
                noise = torch.randn_like(latents)
                noise_timesteps = torch.tensor(noise_timesteps).long()   
                latents = noise_scheduler.add_noise(latents, noise, noise_timesteps)
        
        if denoise_timesteps > 0:        
            cond, uncond = get_condition(batch, use_cond=True), get_condition(batch, use_cond=False)
            
            # # DDIM
            noise_scheduler.config.num_train_timesteps=denoise_timesteps
            noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        
            for _, t in enumerate(noise_scheduler.timesteps):
                t_batch = torch.tensor([t] * latents.shape[0], device=latents.device)
                noise_pred_uncond, noise_pred_cond = unet(latents, t_batch, **uncond)[0], unet(latents, t_batch, **cond)[0]
                noise_pred = noise_pred_uncond + 2 * (noise_pred_cond - noise_pred_uncond)
                classifier_gradient = get_classifier_gradient(latents, **batch) if use_classifier_guidence else None
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False, classifier_gradient=classifier_gradient)[0]
                        
        # get detection results
        latents = latents.permute(0, 2, 3, 1)            
        latents = latents.reshape(-1, bev_cfg.bev_h_*bev_cfg.bev_w_, bev_cfg._dim_)
        det_result = bev_model(return_loss=False, only_bev=False, given_bev=latents, rescale=True, **batch)
        
        if isinstance(det_result, dict):
            if 'bbox_results' in det_result.keys():
                bbox_result = det_result['bbox_results']
                batch_size = len(det_result['bbox_results'])
                bbox_results.extend(bbox_result)
            if 'mask_results' in det_result.keys() and det_result['mask_results'] is not None:
                mask_result = custom_encode_mask_results(det_result['mask_results'])
                mask_results.extend(mask_result)
                have_mask = True
        else:
            batch_size = len(det_result)
            bbox_results.extend(det_result)
            
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir=os.path.join(save_path, '.dist_test'))
    if have_mask:
        mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir=os.path.join(save_path, '.dist_test'))
    else:
        mask_results = None
    
    det_results = bbox_results if mask_results is None else {'bbox_results': bbox_results, 'mask_results': mask_results}
    
    key_score = {}
    if rank == 0:
        eval_kwargs = bev_cfg.get('evaluation', {}).copy()
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        eval_kwargs['jsonfile_prefix'] = os.path.join(save_path, det_res_path)
        eval_results = dataset.evaluate(det_results, **eval_kwargs)
        for metric, score in eval_results.items():
            if 'mAP' in  metric or 'NDS' in metric:
                key_score[metric] = score
    return key_score       
  

if __name__ == "__main__":
    test()





