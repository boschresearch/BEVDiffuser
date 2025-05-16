# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.parallel import DataContainer as DC
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.nuscenes_dataset_v2 import CustomNuScenesDatasetV2
 
@DATASETS.register_module()
class CustomNuScenesDiffusionDataset_layout(CustomNuScenesDataset): 
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12, num_bboxes=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.object_names = list(self.CLASSES) + ['__image__', '__null__']
        self.object_clips = self.embed_object_names()
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx) 
        layout = self.get_layout_info(data)
        for info in layout.keys():
            data[info] = layout[info]
        return data
    
    def embed_object_names(self):
        pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-1'
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        object_tokens = tokenizer(self.object_names, 
                                  max_length=tokenizer.model_max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt").input_ids
        object_clip_embed = text_encoder(object_tokens)[1]
        return object_clip_embed
    
    def get_layout_info(self, data):
        # data['gt_labels_3d'] should be a DC of a tensor with size [N]
        # data['gt_bboxes_3d'] should be a DC of LiDARInstance3DBoxes of a tensor with size [N, 9]: (x, y, z, x_size, y_size, z_size, yaw, vx, vy) 
        class_ids = torch.tensor([])
        gt_bboxes = torch.tensor([])
        if 'gt_labels_3d' in data:
            while not isinstance(data['gt_labels_3d'], DC):
                data['gt_labels_3d'] = data['gt_labels_3d'][0]
            class_ids = data['gt_labels_3d'].data
            class_ids = (class_ids + len(self.CLASSES))%len(self.CLASSES)
        if 'gt_bboxes_3d' in data:
            while not isinstance(data['gt_bboxes_3d'], DC):
                data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
            gt_bboxes = data['gt_bboxes_3d'].data
            
        layout_obj_classes = torch.LongTensor(self.num_bboxes).fill_(self.num_classes-1)
        layout_is_valid = torch.zeros([self.num_bboxes])
        
        layout_obj_classes[0] = self.num_classes-2
        layout_is_valid[0] = 1.0
        default_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
        
        num_valid = min(len(class_ids), self.num_bboxes-1)
        layout_obj_classes[1: 1+num_valid] = class_ids
        layout_is_valid[1: 1+num_valid] = 1.0
        
        layout_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
       
        
        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_bboxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_bboxes)
            
        layout = {
            'layout_obj_classes': DC(layout_obj_classes),
            'layout_obj_bboxes': DC(layout_obj_bboxes),
            'layout_obj_is_valid': DC(layout_is_valid),
            'layout_obj_names': DC(layout_obj_clip),
            'default_obj_names': DC(default_obj_clip),
        }

        return layout
    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
            
            
@DATASETS.register_module()
class CustomNuScenesDiffusionDatasetV2_layout(CustomNuScenesDatasetV2): 
    def __init__(self, pc_range, use_3d_bbox=True, num_classes=12, num_bboxes=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_range = pc_range
        self.use_3d_bbox=use_3d_bbox
        self.num_classes=num_classes
        self.num_bboxes=num_bboxes
        self.object_names = list(self.CLASSES) + ['__image__', '__null__']
        self.object_clips = self.embed_object_names()
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx) 
        layout = self.get_layout_info(data)
        for info in layout.keys():
            data[info] = layout[info]
        return data
    
    def embed_object_names(self):
        pretrained_model_name_or_path = 'stabilityai/stable-diffusion-2-1'
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        text_encoder.requires_grad_(False)
        object_tokens = tokenizer(self.object_names, 
                                  max_length=tokenizer.model_max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_tensors="pt").input_ids
        object_clip_embed = text_encoder(object_tokens)[1]
        return object_clip_embed
    
    def get_layout_info(self, data):
        # data['gt_labels_3d'] should be a DC of a tensor with size [N]
        # data['gt_bboxes_3d'] should be a DC of LiDARInstance3DBoxes of a tensor with size [N, 9]: (x, y, z, x_size, y_size, z_size, yaw, vx, vy) 
        class_ids = torch.tensor([])
        gt_bboxes = torch.tensor([])
        if 'gt_labels_3d' in data:
            while not isinstance(data['gt_labels_3d'], DC):
                data['gt_labels_3d'] = data['gt_labels_3d'][0]
            class_ids = data['gt_labels_3d'].data
            class_ids = (class_ids + len(self.CLASSES))%len(self.CLASSES)
        if 'gt_bboxes_3d' in data:
            while not isinstance(data['gt_bboxes_3d'], DC):
                data['gt_bboxes_3d'] = data['gt_bboxes_3d'][0]
            gt_bboxes = data['gt_bboxes_3d'].data
            
        layout_obj_classes = torch.LongTensor(self.num_bboxes).fill_(self.num_classes-1)
        layout_is_valid = torch.zeros([self.num_bboxes])
        
        layout_obj_classes[0] = self.num_classes-2
        layout_is_valid[0] = 1.0
        default_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
        
        num_valid = min(len(class_ids), self.num_bboxes-1)
        layout_obj_classes[1: 1+num_valid] = class_ids
        layout_is_valid[1: 1+num_valid] = 1.0
        
        layout_obj_clip = torch.stack([self.object_clips[int(cid)] for cid in layout_obj_classes])
       
        
        if self.use_3d_bbox:
            layout_obj_bboxes = self.get_3d_layout_bboxes(gt_bboxes)
        else:
            layout_obj_bboxes = self.get_2d_layout_bboxes(gt_bboxes)
            
        layout = {
            'layout_obj_classes': DC(layout_obj_classes),
            'layout_obj_bboxes': DC(layout_obj_bboxes),
            'layout_obj_is_valid': DC(layout_is_valid),
            'layout_obj_names': DC(layout_obj_clip),
            'default_obj_names': DC(default_obj_clip),
        }

        return layout
    
    def normalize_bbox(self, bbox):
        # normalize bbox into [0,1], ego at [0.5, 0.5] 
        x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
        x = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        if bbox.shape[-1] > 2:
            z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(bbox[..., 2:], 7, dim=-1)
            z = (z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
            x_size = x_size / (self.pc_range[3] - self.pc_range[0])
            y_size = y_size / (self.pc_range[4] - self.pc_range[1])
            z_size = z_size / (self.pc_range[5] - self.pc_range[2])
            return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy), dim=-1)
        return torch.cat((x, y), dim=-1)
            
    
    def get_3d_layout_bboxes(self, gt_bboxes): 
        # 3d bbox: (xc, yc, zc, x_size, y_size, z_size, yaw, vx, vy) 
        # ego coordinate, origin at ego position, x towards right, y towards front 
        layout_bboxes = torch.zeros([self.num_bboxes, 9])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 0, 1, 1, 1, 0, 0, 0])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            # (x, y) -> (x-0.5, x-0.5)
            gt_bboxes = self.normalize_bbox(gt_bboxes.tensor)
            gt_bboxes[..., :2] = gt_bboxes[..., :2] - 0.5
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
    
    def get_2d_layout_bboxes(self, gt_bboxes):
        # 2d bbox: (x0, y0, x1, y1), 
        # image coordinate, orgin at upper left, x towards right, y towards down
        layout_bboxes = torch.zeros([self.num_bboxes, 4])
        layout_bboxes[0] = torch.FloatTensor([0, 0, 1, 1])
        if isinstance(gt_bboxes, LiDARInstance3DBoxes):
            gt_bboxes = self.normalize_bbox(gt_bboxes.corners[..., :2]) # N x 8 x 2
            # (x, y) -> (x, 1-y)
            gt_bboxes[..., 1] = 1 - gt_bboxes[..., 1]
            gt_bboxes_min = gt_bboxes.min(dim=1).values # N x 2
            gt_bboxes_max = gt_bboxes.max(dim=1).values # N x 2
            gt_bboxes = torch.cat((gt_bboxes_min, gt_bboxes_max), dim=-1)
            num_valid = min(len(gt_bboxes), self.num_bboxes-1)
            layout_bboxes[1: 1+num_valid] = gt_bboxes
        return layout_bboxes
             