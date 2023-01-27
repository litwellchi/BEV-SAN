# Copyright (c) Megvii Inc. All rights reserved.
from argparse import ArgumentParser, Namespace

import mmcv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR
import copy
import os

from dataset.nusc_mv_det_dataset import NuscMVDetDataset, collate_fn
from evaluators.det_mv_evaluators import DetMVNuscEvaluator
# from models.bev_depth import BEVDepth
from models.uda_depth import BEVDepth
from layers.discriminator.img_disc import Disc_img_source
from layers.discriminator.img_disc import Disc_img_target
from layers.discriminator.bev_disc import Disc_bev_source
from layers.discriminator.bev_disc import Disc_bev_target
# from layers.discriminator.vox_disc import Disc_vox_source
# from layers.discriminator.vox_disc import Disc_vox_target
from layers.discriminator.cam_disc import Disc_cam_source
from layers.discriminator.cam_disc import Disc_cam_target
from utils.torch_dist import all_gather_object, get_rank, synchronize
from layers.reverse_layer import ReverseLayerF
from ops.adam import Adam


H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


class BEVDepthLightningModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 **kwargs):
        super().__init__()

        # disable auto optimization
        self.automatic_optimization = False

        self.save_hyperparameters()
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.basic_lr_per_img = 2e-6 / 64
        self.class_names = class_names
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        mmcv.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetMVNuscEvaluator(class_names=self.class_names,
                                            output_dir=self.default_root_dir)
        self.teacher = BEVDepth(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)
        self.student = BEVDepth(self.backbone_conf,
                              self.head_conf,
                              is_train_depth=True)
        self.model = self.student
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_depth = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])

        self.disc_img_source = Disc_img_source()
        self.disc_img_target = Disc_img_target()
        self.disc_bev_source = Disc_bev_source()
        self.disc_bev_target = Disc_bev_target()
        self.disc_cam_source = Disc_cam_source()
        self.disc_cam_target = Disc_cam_target()
        # self.disc_vox_source = Disc_vox_source()
        # self.disc_vox_target = Disc_vox_target()
        self.img_source_loss = torch.nn.NLLLoss()
        self.img_target_loss = torch.nn.NLLLoss()
        # self.vox_source_loss = torch.nn.NLLLoss()
        # self.vox_target_loss = torch.nn.NLLLoss()
        self.cam_source_loss = torch.nn.NLLLoss()
        self.cam_target_loss = torch.nn.NLLLoss()
        # patch gan
        self.bev_source_loss = torch.nn.BCEWithLogitsLoss() 
        self.bev_target_loss = torch.nn.BCEWithLogitsLoss() 

        for param in self.teacher.parameters():
            param.detach_()

    def forward(self, sweep_imgs, mats, depth_label=None):
        return self.model(sweep_imgs, mats, depth_label)

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def make_sudo_label(self, preds, img_metas, threshold=0.7):
        sudo_boxes, sudo_labels = [],[]
        result = self.model.get_bboxes(preds, img_metas)
        for i in range(len(result)):
            sudo_box,sudo_label = [],[]
            box = result[i][0].tensor.detach().cpu().numpy()
            score = result[i][1].detach().cpu().numpy()
            label = result[i][2].detach().cpu().numpy()
            for j in range(len(label)):
                if score[j]>threshold:
                    sudo_box.append(box[j])
                    sudo_label.append(label[j])
            sudo_boxes.append(torch.tensor(sudo_box).cuda())
            sudo_labels.append(torch.tensor(sudo_label).cuda())
        return sudo_boxes,sudo_labels

    def training_step(self, batch, batch_idx):
        # get optimizer
        opt, opt_img_source, opt_img_target, opt_bev_source, opt_bev_target, opt_cam_source, opt_cam_target = self.optimizers()
    #     # opt, opt_img_source, opt_img_target, opt_bev_source, opt_bev_target = self.optimizers()
        
        opt.zero_grad()
        opt_img_source.zero_grad()
        opt_img_target.zero_grad()
        opt_bev_source.zero_grad()
        opt_bev_target.zero_grad()
        opt_cam_source.zero_grad()
        opt_cam_source.zero_grad()
        # opt_vox_source.zero_grad()
        # opt_vox_source.zero_grad()

        # training model using source data**************************************    
        (sweep_imgs, mats, a, b, gt_boxes, gt_labels, depth_labels) = batch["source_train_loader"]
        preds, depth_preds, img_feats_source, bev_feats_source, vox_feats_source, cam_feats_source = self.student(sweep_imgs, mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)
        if len(depth_labels.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        self.log('Source detection_loss', detection_loss)
        self.log('Source depth_loss', depth_loss)

        # Geting sudo label of target data using teacher net ***************************************    
        (sweep_imgs_target, mats_target, _, img_metas ,gt_boxes_target, gt_labels_target, depth_labels_target) = batch["target_train_loader"]
        if torch.cuda.is_available():
            for key, value in mats_target.items():
                mats_target[key] = value.cuda()
            sweep_imgs_target = sweep_imgs_target.cuda()

        downsampled_depth_labels_target = self.get_downsampled_gt_depth(depth_labels_target,same_as_pred=True)
        with torch.no_grad():
            preds_target_teacher, depth_preds_target_teacher, img_feats_teacher, bev_feats_teacher, vox_feats_teacher, cam_feats_teacher = self.teacher(sweep_imgs_target, mats_target, depth_label=downsampled_depth_labels_target)

        preds_target, depth_preds_target, img_feats_target, bev_feats_target, vox_feats_target, cam_feats_target = self.student(sweep_imgs_target, mats_target)

        # test depth gt improve ideas
        # # downsampled_depth_labels_target + depth_preds_target
        # depth_empty_space = (downsampled_depth_labels_target==0.)
        # count_zeros = torch.sum(depth_empty_space)
        # sudo_depth = downsampled_depth_labels_target + torch.mul(depth_empty_space,depth_preds_target)
        # # depth_preds_target
        # count_zeros2 = torch.sum(sudo_depth)
        
        # Student loss
        sudo_boxes,sudo_labels = self.make_sudo_label(preds_target_teacher,img_metas, threshold=0.3)
        sudo_depth = depth_preds_target_teacher
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            sudo_tar_target = self.model.module.get_targets(sudo_boxes, sudo_labels)
            detection_loss_target = self.model.module.loss(sudo_tar_target, preds_target)
        else:
            sudo_tar_target = self.model.get_targets(sudo_boxes, sudo_labels)
            detection_loss_target = self.model.loss(sudo_tar_target, preds_target)
        if len(depth_labels_target.shape) == 5:
            # only key-frame will calculate depth loss
            depth_labels_target = depth_labels_target[:, 0, ...]
        depth_loss_target = self.get_depth_loss(depth_labels_target.cuda(), depth_preds_target)
        self.log('target detection_loss', detection_loss_target)
        self.log('target depth_loss', depth_loss_target)

        # Add heads on student model
      # preparing source pseudo domain label   
        domain_label_source = torch.zeros(len(sweep_imgs), requires_grad=True)
        domain_label_source = domain_label_source.long().cuda()
        cam_domain_label_source = torch.zeros(len(sweep_imgs)*6, requires_grad=True)
        cam_domain_label_source = cam_domain_label_source.long().cuda()
        # vox_domain_label_source = torch.zeros(len(sweep_imgs)*6, requires_grad=True)
        # vox_domain_label_source = vox_domain_label_source.long().cuda()
        # patch gan
        bev_domain_label_source = torch.zeros(len(sweep_imgs), 1, requires_grad=True)
        bev_domain_label_source = bev_domain_label_source.float().cuda()

      # preparing target pseudo domain label 
        domain_label_target = torch.ones(len(sweep_imgs_target), requires_grad=True)
        domain_label_target = domain_label_target.long().cuda()
        cam_domain_label_target = torch.ones(len(sweep_imgs_target)*6, requires_grad=True)
        cam_domain_label_target = cam_domain_label_target.long().cuda()
        # vox_domain_label_target = torch.ones(len(sweep_imgs_target)*6, requires_grad=True)
        # vox_domain_label_target = vox_domain_label_target.long().cuda()
        # patch gan
        bev_domain_label_target = torch.ones(len(sweep_imgs_target), 1, requires_grad=True)
        bev_domain_label_target = bev_domain_label_target.float().cuda()

      # get reverse feature for img and bev feature
        reverse_img_4_source = ReverseLayerF.apply(img_feats_source, 1)
        reverse_img_4_target = ReverseLayerF.apply(img_feats_target, 1)
        reverse_bev_4_source = ReverseLayerF.apply(bev_feats_source, 1)
        reverse_bev_4_target = ReverseLayerF.apply(bev_feats_target, 1)
        reverse_cam_4_source = ReverseLayerF.apply(cam_feats_source, 1)
        reverse_cam_4_target = ReverseLayerF.apply(cam_feats_target, 1)
        # reverse_vox_4_source = ReverseLayerF.apply(vox_feats_source, 1)
        # reverse_vox_4_target = ReverseLayerF.apply(vox_feats_target, 1)

        # get discriminaotr output for img and bev feature
        img_out_4_source = self.disc_img_source(reverse_img_4_source)
        img_out_4_target = self.disc_img_target(reverse_img_4_target)
        bev_out_4_source = self.disc_bev_source(reverse_bev_4_source)
        bev_out_4_target = self.disc_bev_target(reverse_bev_4_target)
        cam_out_4_source = self.disc_cam_source(reverse_cam_4_source)
        cam_out_4_target = self.disc_cam_target(reverse_cam_4_target)
        # vox_out_4_source = self.disc_vox_source(reverse_vox_4_source)
        # vox_out_4_target = self.disc_vox_target(reverse_vox_4_target)

        # adv loss
        source_img_adv_loss = self.img_source_loss(img_out_4_source, domain_label_source)
        target_img_adv_loss = self.img_target_loss(img_out_4_target, domain_label_target)
        source_bev_adv_loss = self.bev_source_loss(bev_out_4_source, bev_domain_label_source)
        target_bev_adv_loss = self.bev_target_loss(bev_out_4_target, bev_domain_label_target)
        source_cam_adv_loss = self.cam_source_loss(cam_out_4_source, cam_domain_label_source)
        target_cam_adv_loss = self.cam_target_loss(cam_out_4_target, cam_domain_label_target)
        # source_vox_adv_loss = self.vox_source_loss(vox_out_4_source, vox_domain_label_source)
        # target_vox_adv_loss = self.vox_source_loss(vox_out_4_target, vox_domain_label_target)

        # loss backward
        overall_loss = detection_loss_target + depth_loss_target + detection_loss + depth_loss + 0.1*(source_img_adv_loss + target_img_adv_loss) + 1*(source_bev_adv_loss + target_bev_adv_loss) + 0.1*(source_cam_adv_loss + target_cam_adv_loss)
        # overall_loss = detection_loss + depth_loss + 0.1*(source_img_adv_loss + target_img_adv_loss) + 1*(source_bev_adv_loss + target_bev_adv_loss)

        # overall_loss = detection_loss_target + depth_loss_target + detection_loss + depth_loss
        self.log_dict({"target psesudo loss": detection_loss_target + depth_loss_target,"source loss": detection_loss + depth_loss})
        # self.log_dict({"loss": overall_loss, "img_loss": source_img_adv_loss+ target_img_adv_loss, "bev_loss": source_bev_adv_loss + target_bev_adv_loss, "det_dep_loss":detection_loss + depth_loss}, prog_bar=True)

        self.manual_backward(loss=overall_loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        # optimizer
        opt_bev_source.step()
        opt_bev_target.step()
        opt_img_source.step()
        opt_img_target.step()
        opt_cam_source.step()
        opt_cam_target.step()
        # opt_vox_source.step()
        # opt_vox_target.step()
        opt.step()

        current_step = self.global_step
        self.update_ema_variables(self.student, self.teacher, alpha=0.999, global_step=self.global_step)



    def training_epoch_end(self, training_step_outputs):
        print("save checkpoint to "+os.path.join(self.default_root_dir,"./save_checkpoint/epoch_{}.ckpt".format(self.current_epoch)))
        self.trainer.save_checkpoint(os.path.join(self.default_root_dir,"./save_checkpoint/epoch_{}.ckpt".format(self.current_epoch)))


    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths,same_as_pred=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        if same_as_pred is true:
            Output:
            gt_depths: [B,n,h,w]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        if same_as_pred:
            gt_depths =  F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1)[:, :, :, 1:]
            # [B,h,w,n] -> [B,n,h,w]
            gt_depths = gt_depths.permute(0,3,1,2)
        else:
            gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        return gt_depths.float()

    def eval_step(self, batch, batch_idx, prefix: str):
        if prefix =='depthval':
            (sweep_imgs, mats, a, img_metas, gt_boxes, gt_labels, depth_labels) = batch
            #  Eval with depth
            downsampled_depth_labels = self.get_downsampled_gt_depth(depth_labels,same_as_pred=True)

            #test code here
            # downsampled_depth_labels_target + depth_preds_target
            _, depth_preds, _, _, _, _ = self.model(sweep_imgs, mats)
            depth_empty_space = (downsampled_depth_labels==0.)
            sudo_depth = downsampled_depth_labels + torch.mul(depth_empty_space,depth_preds)
            # depth_preds_target new pseudo
            preds, _, _, _, _, _ = self.model(sweep_imgs, mats, depth_label=sudo_depth)

        else:
            (sweep_imgs, mats, _, img_metas, _, _, depth_labels) = batch

            if torch.cuda.is_available():
                for key, value in mats.items():
                    mats[key] = value.cuda()
                sweep_imgs = sweep_imgs.cuda()
            #  Normal eval
            preds = self.model(sweep_imgs, mats)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].tensor.detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def test_epoch_end(self, test_step_outputs):
        all_pred_results = list() 
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-5)
        optimizer_img_source = torch.optim.Adam(self.disc_img_source.parameters(), lr=1e-5)
        optimizer_img_target = torch.optim.Adam(self.disc_img_target.parameters(), lr=1e-5)
        optimizer_bev_source = torch.optim.Adam(self.disc_bev_source.parameters(), lr=1e-5)
        optimizer_bev_target = torch.optim.Adam(self.disc_bev_target.parameters(), lr=1e-5)
        optimizer_cam_source = torch.optim.Adam(self.disc_cam_source.parameters(), lr=1e-5)
        optimizer_cam_target = torch.optim.Adam(self.disc_cam_target.parameters(), lr=1e-5)
        # optimizer_vox_source = torch.optim.Adam(self.disc_vox_source.parameters(), lr=1e-5)
        # optimizer_vox_target = torch.optim.Adam(self.disc_vox_target.parameters(), lr=1e-5)
        scheduler = MultiStepLR(optimizer, [1,2,3,4,5,6,7])

        # return [optimizer]
        # return [[optimizer], [scheduler]]
        # return [[optimizer, optimizer_img_source, optimizer_img_target, optimizer_bev_source, optimizer_bev_target], [scheduler]]
        return [optimizer, optimizer_img_source, optimizer_img_target, optimizer_bev_source, optimizer_bev_target, optimizer_cam_source, optimizer_cam_target]

    def train_dataloader(self):
        train_source_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            # info_path='data/nuScenes/nuscenes_12hz_infos_train.pkl',
            info_path='/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_train_boston.pkl',
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=self.data_return_depth,
        )
        from functools import partial

        train_target_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            # info_path='data/nuScenes/nuscenes_12hz_infos_train.pkl',
            info_path='/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_train_singapore.pkl',
            # info_path = "/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_val_singapore.pkl",
            is_train=True,
            use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=self.data_return_depth,
        )
        from functools import partial

        train_source_loader = torch.utils.data.DataLoader(
            train_source_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth),
            sampler=None,
        )

        train_target_loader = torch.utils.data.DataLoader(
            train_target_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth),
            sampler=None,
        )

        train_loader = {"source_train_loader": train_source_loader, "target_train_loader": train_target_loader}

        return train_loader
        # return train_target_loader

    def val_dataloader(self):
        val_source_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            # info_path='data/nuScenes/nuscenes_12hz_infos_val.pkl',
            info_path='/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_val_boston.pkl',
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=False,
        )

        val_target_dataset = NuscMVDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            # info_path='data/nuScenes/nuscenes_12hz_infos_val.pkl',
            info_path='/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_val_singapore.pkl',
            # info_path='/home/notebook/data/group/zhangrongyu/code/BEVDepth/data/nuScenes/nuscenes_12hz_infos_train_singapore.pkl',
            is_train=False,
            img_conf=self.img_conf,
            num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes,
            key_idxes=self.key_idxes,
            return_depth=True,
        )      

        val_source_loader = torch.utils.data.DataLoader(
            val_source_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            sampler=None,
        )

        from functools import partial
        val_target_loader = torch.utils.data.DataLoader(
            val_target_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth),
            num_workers=4,
            sampler=None,
        )
        
        val_loader = {"source_val_loader": val_source_loader, "target_val_loader": val_target_loader}

        return val_target_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'depthval')
        # return self.eval_step(batch, batch_idx, 'test')

    def reload_hyperparameter(self, gpus: int = 1,
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 default_root_dir='./outputs/',
                 **kwargs):
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.data_root = data_root
        self.default_root_dir=default_root_dir
        self.data_return_depth = True

    def reload_teacherNet(self, teacher_model):
        self.teacher = teacher_model
    def reload_studentNet(self, student_model):
        self.student = student_model
        self.model = self.student

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    # if args.ckpt_path is None:
    #    model = BEVDepthLightningModel(**vars(args))
    # Freeze train of the teacher model
    if args.teacher_ckpt is not None:
        # teacher_model = BEVDepthLightningModel(**vars(args)).load_from_checkpoint(args.teacher_ckpt, strict=False)
        # student_model = BEVDepthLightningModel(**vars(args))
        # student_model.reload_teacherNet(teacher_model.model)
        # model = student_model
        # model.reload_hyperparameter(**vars(args))
        # # Both load pretrain
        teacher_model = BEVDepthLightningModel(**vars(args)).load_from_checkpoint(args.teacher_ckpt, strict=False)
        student_model = BEVDepthLightningModel(**vars(args)).load_from_checkpoint(args.teacher_ckpt, strict=False)
        student_model.reload_teacherNet(teacher_model.model)
        student_model.reload_studentNet(student_model.model)
        model = student_model
        model.reload_hyperparameter(**vars(args))
    else:
        model = BEVDepthLightningModel(**vars(args)).load_from_checkpoint(args.ckpt_path, strict=False)
        model.reload_hyperparameter(**vars(args))

    # val_dataloader = model.val_dataloader()
    # eval = model.custom_eval(val_dataloader)
    
    trainer = pl.Trainer.from_argparse_args(args)
    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parent_parser.add_argument('--teacher_ckpt', type=str)
    parser = BEVDepthLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=10,
        accelerator='ddp',
        num_sanity_val_steps=0,
        # gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=False,
        precision=16,
        default_root_dir='./outputs/bevuda-b2s&tshead&ct03-ema0001-wdecay')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
