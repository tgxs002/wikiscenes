from __future__ import print_function


import os, json, random, sys, math, torch, copy, hashlib, io
torch.manual_seed(1)
random.seed(1)
import numpy as np
np.random.seed(1)
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix
import pandas
import _pickle as cPickle

from datasets import get_dataloader, get_num_classes, get_class_names
from models import get_model

from base_trainer import BaseTrainer
from functools import partial

from opts import get_arguments
from core.config import cfg, cfg_from_file, cfg_from_list
from datasets.utils import Colorize
from losses import get_criterion, mask_loss_ce

from utils.timer import Timer
from utils.stat_manager import StatManager
from torchvision.utils import save_image as sv
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map
from PIL import Image, ImagePalette

# specific to pytorch-v1 cuda-9.0
# see: https://github.com/pytorch/pytorch/issues/15054#issuecomment-450191923
# and: https://github.com/pytorch/pytorch/issues/14456
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
DEBUG = False
import cv2

# overlay image
def overlay_map_on_im(im, predmap, point, overlay_path, display=False, weight_ratio=[0.5,0.5]):
    im = im.permute(1, 2, 0).cpu().numpy()*255
    predmap = predmap.cpu().numpy()
    predmap = cv2.resize(predmap, (im.shape[1], im.shape[0]))

    predmap_dup = np.stack((255*predmap,)*3, axis=-1)
    predmap_dup[:,:,0] = 0
    predmap_dup[:,:,1] = 0
    # predmap_dup[:,:,2] = 0
    overlay = cv2.addWeighted(im, weight_ratio[0], predmap_dup, weight_ratio[1], 0)
    # draw a cross
    green = (250, 250, 250)
    mark_size = 5
    x = point[0]
    y = point[1]
    for k in range(x - mark_size, x + 1 + mark_size):
        if 0 <= k < overlay.shape[0]:
            overlay = cv2.circle(overlay, (k, y), radius=0, color=green, thickness=-1)
    for k in range(y - mark_size, y + 1 + mark_size):
        if 0 <= k < overlay.shape[1]:
            overlay = cv2.circle(overlay, (x, k), radius=0, color=green, thickness=-1)
    # overlay = cv2.circle(overlay, point, radius=5, color=(0, 0, 0), thickness=-1) #gbr
    cv2.imwrite(overlay_path, cv2.cvtColor(cv2.hconcat([im,overlay]), cv2.COLOR_BGR2RGB))

def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(org_im.shape[1:], Image.ANTIALIAS)
    org_im = transforms.ToPILImage()(org_im).convert("RGBA")
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image

def rescale_as(x, y, mode="bilinear", align_corners=True):
    h, w = y.size()[2:]
    x = F.interpolate(x, size=[h, w], mode=mode, align_corners=align_corners)
    return x

class DecTrainer(BaseTrainer):

    def __init__(self, args, **kwargs):
        super(DecTrainer, self).__init__(args, **kwargs)

        # dataloader
        self.trainloader = get_dataloader(args, cfg, 'train')
        self.export_task = args.export
        self.export_set = args.export_set
        if self.export_task == "None" or self.export_set == "val":
            self.valloader = get_dataloader(args, cfg, 'val')
        else:
            self.valloader = None

        if self.export_task == "None" or self.export_set == "val_easy":
            self.valloader_easy = get_dataloader(args, cfg, "val_easy")
        else:
            self.valloader_easy = None

        self.denorm = self.trainloader.dataset.denorm
        self.use_triplet = args.use_triplet
        self.loss_3d = args.loss_3d
        self.normalize_feature = args.normalize_feature
        self.feature_layer = args.feature
        self.run = args.run
        self.multi_label = args.multi_label
        self.use_contrastive = args.use_contrastive
        self.use_contrastive_easy = args.use_contrastive_easy
        self.pretrain = args.pretrain
        self.lr_milestones = [int(x) for x in args.lr_milestones]

        assert not self.use_triplet or not self.use_contrastive, "use triplet loss or contrastive loss?"
        if self.use_contrastive:
            self.num_negative = args.num_negative
            assert self.num_negative > 1, "please sample more than 1 negative point"
            self.tau = args.contrastive_tau
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss().cuda()

        self.nclass = get_num_classes(args)
        self.classNames = get_class_names(args)
        assert self.nclass == len(self.classNames) - 1

        self.classIndex = {}
        for i, cname in enumerate(self.classNames):
            self.classIndex[cname] = i

        # model
        self.enc = get_model(cfg.NET, num_classes=self.nclass)
        self.criterion_cls = get_criterion(cfg.NET.LOSS)

        # optimizer using different LR
        enc_params = self.enc.parameter_groups(cfg.NET.LR, cfg.NET.WEIGHT_DECAY)
        self.optim_enc = self.get_optim(enc_params, cfg.NET)

        # checkpoint management
        self._define_checkpoint('enc', self.enc, self.optim_enc)

        # using cuda
        if cfg.NUM_GPUS != 0:
            self.enc = nn.DataParallel(self.enc)
            self.criterion_cls = nn.DataParallel(self.criterion_cls)
            self.enc = self.enc.cuda()
            self.criterion_cls = self.criterion_cls.cuda()

        self._load_checkpoint(args.resume)

        # lr decay
        self.scheduler = MultiStepLR(self.optim_enc, milestones=self.lr_milestones, gamma=0.1, last_epoch=self.start_epoch-1)

        self.fixed_batch = None
        self.fixed_batch_path = args.fixed_batch_path
        if os.path.isfile(self.fixed_batch_path):
            print("Loading fixed batch from {}".format(self.fixed_batch_path))
            self.fixed_batch = torch.load(self.fixed_batch_path)

        self.visual_times = 0
        self.dataset = args.dataset.lower()

    def step(self, epoch, image, gt_labels, train=False, visualise=False, save_image=False, info=None, debug=False):

        assert not debug or train, "If you want to visualize the pairs, please do this during training, otherwise make sure both query images and matching images are passed to step(), and then comment this line"

        PRETRAIN = epoch < (11 if DEBUG else cfg.TRAIN.PRETRAIN)

        if self.dataset in ["wikiscenes_corr", "wikiscenes_corr_semi"]:
            corr = image["corr"]
            image = image["image"] # images are organized in the batch such that indices b and b + bs // 2 are pairs.
        
        # denorm image
        image_raw = self.denorm(image.clone())  #[32, 3, 224, 224], vals range in [0,1]

        # classification
        cls_out, cls_fg, masks, mask_logits, pseudo_gt, loss_mask = self.enc(image, image_raw, gt_labels, self.feature_layer)

        # classification loss
        if self.dataset == "wikiscenes_corr_semi":
            bs = cls_out.shape[0] // 2
            loss_cls = self.criterion_cls(cls_out[:bs], gt_labels[:bs]).mean()
            cls_fg = cls_fg[:bs]
            loss_mask = loss_mask[:bs]
        else:
            loss_cls = self.criterion_cls(cls_out, gt_labels).mean()

        # keep track of all losses for logging
        losses = {"loss_cls": loss_cls.item()}
        losses["loss_fg"] = cls_fg.mean().item()

        loss_3d = 0
        mean_3d_loss = 0
        kp_number = 0
        norm = 0
        if self.dataset in ["wikiscenes_corr", "wikiscenes_corr_semi"] and train:
            # compute 3D consistency loss
            feature = masks["feature"]   #[32, 256, 56, 56]
            # here we normalize the out_feature to make sure it doesn't optimize by scaling the feature vector

            b, c, h, w = feature.shape
            feature = feature.reshape(2, b // 2, c, h, w)  #[2, 16, 256, 56, 56]
            assert h == w, "not square"

            # modify feature so that it has the same batch size as coord
            if self.use_triplet:
                modified_feature = torch.cat([feature, feature[1:]])
            elif self.use_contrastive and not self.use_contrastive_easy:
                modified_feature = torch.cat([feature, feature[1:].repeat(self.num_negative,1,1,1,1)])  #[12, 16, 256, 56, 56]
            else:
                modified_feature = feature

            if debug:
                coord_arr = list()  # keep for visualization purposes
            for i in range(b // 2):
                if not corr[i]:
                    if debug:
                        coord_arr.append(torch.Tensor(1))
                    continue
                # k * 4
                p = corr[i].values()  # [y0,x0,y1,x1]
                coord = torch.tensor(list(p))  # [24,4]
                if cfg.NUM_GPUS != 0:
                    coord = coord.cuda()
                # reshape, permute to allow grid_sample, [N, Hout, Wout, 2]
                coord = coord.reshape(1, -1, 2, 2).permute(2, 0, 1, 3).contiguous()  # [2, 1, 24, 2]
                if self.use_contrastive_easy:
                    perm_rand = torch.randint(b // 2, (self.num_negative,))
                    modified_feature = torch.cat((feature[0,i].unsqueeze(0),
                                                  feature[1,i].unsqueeze(0),
                                                  feature[1,perm_rand]))
                    coord = torch.cat([coord, coord[1:2].repeat(self.num_negative,1,1,1)])
                    coord[2:, 0, :, :] = torch.rand(self.num_negative, coord.shape[2], 2).cuda()
                elif self.use_triplet:
                    # add a fake kp
                    coord = torch.cat([coord, coord[1:2]])
                    # add a random shift in [0.25, 0.75], fmod it
                    # so the selected fake keypoint is far enough from the correct corresponding one

                    ### 1. previous sampling method
                    # coord[2,0,:,:] += (torch.rand(coord.shape[2], 2) / 2 + 0.25).cuda()
                    # coord = coord.fmod(1.0)
                    ### close 1

                    ### 2. just randomly sample some points
                    coord[2,0,:,:] += (torch.rand(coord.shape[2], 2)).cuda()
                    ### close 2

                elif self.use_contrastive:
                    # sample negative
                    coord = torch.cat([coord, coord[1:2].repeat(self.num_negative,1,1,1)])  #[12, 1, 24, 2]
                    # add a random shift in +/-[0.25, 0.75]
                    # so the selected fake keypoint is far enough from the correct corresponding one

                    ### 1. previous version
                    # coord[2:,0,:,:] += (torch.rand(self.num_negative, coord.shape[2], 2) / 2 + 0.25).cuda()
                    # coord = coord.fmod(1.0)   # coord in range [0,1]
                    ### close 1

                    ### 2. just randomly sample some points
                    # coord[2:,0,:,:] += (torch.rand(self.num_negative, coord.shape[2], 2)).cuda()
                    # coord = coord.fmod(1.0)   # coord in range [0,1]
                    ### close 2

                    ### 3. hadar's sampling stragety
                    coord[2:2 + self.num_negative//2, 0, :, 0:1] += ((2*((torch.rand(self.num_negative // 2, coord.shape[2], 1) > 0.5).type(torch.FloatTensor) - 0.5))*(torch.rand(self.num_negative // 2, coord.shape[2], 1) / 2 + 0.25)).cuda()
                    coord[2:2 + self.num_negative//2, 0, :, 1:] =  (torch.rand(self.num_negative // 2, coord.shape[2], 1)).cuda()
                    coord[2 + self.num_negative//2:, 0, :, 1:] += ((2*((torch.rand(self.num_negative // 2, coord.shape[2], 1) > 0.5).type(torch.FloatTensor) - 0.5))*(torch.rand(self.num_negative // 2, coord.shape[2], 1) / 2 + 0.25)).cuda()
                    coord[2 + self.num_negative//2:, 0, :, 0:1] =  (torch.rand(self.num_negative // 2, coord.shape[2], 1)).cuda()
                    coord.fmod(1.0)
                    coord[coord < 0] = 0.0  # coord in range [0,1]
                    ### close 3

                if debug:
                    coord_arr.append(coord.clone())

                # change range to [-1, 1] for grid_sample function call
                coord = coord * 2 - 1
                if self.use_contrastive_easy:
                    keypoints = F.grid_sample(modified_feature, torch.flip(coord, [-1]))
                else:
                    keypoints = F.grid_sample(modified_feature[:,i,:,:], torch.flip(coord, [-1]))  # [12, 256, 1, 24] -- 12: num samples, 256: channels, 24:number_of_keypoints
                loss_func = nn.MSELoss()
                with torch.no_grad():
                    mean_3d_loss += loss_func(feature[:,0,:,:], feature[:,1,:,:])
                    norm += torch.norm(feature).item()
                if self.normalize_feature:
                    keypoints = keypoints / (keypoints.norm(dim=1, keepdim=True) + 1e-6)
                if self.use_triplet:
                    distance_p = (keypoints[0] - keypoints[1]).norm(dim=0)
                    distance_n = (keypoints[0] - keypoints[2]).norm(dim=0)
                    loss_3d += nn.ReLU()(args.triplet_margin + distance_p - distance_n).mean()
                elif self.use_contrastive:
                    # diff = (keypoints[0] - keypoints[1:]) #[11, 256, 1, 24]
                    # distance = (1 - (diff * diff).sum(dim=[1,2]).permute(1,0).contiguous() / 2) / self.tau  # [24, 11]
                    # loss_3d += self.cross_entropy_loss(distance, torch.zeros_like(distance[:,0], dtype=torch.long))
                    num_points = keypoints.shape[-1]
                    if num_points > 1:
                        f_q = keypoints[0].squeeze().repeat(self.num_negative+1,1,1).transpose(0,2) # [24, 256, 11]
                        f_k = keypoints[1:].squeeze().transpose(0,2)
                        input = (f_k * f_q).sum(dim=1) / self.tau  # [24,11]
                        loss_3d += self.cross_entropy_loss(input, torch.zeros_like(input[:,0], dtype=torch.long))
                else:
                    loss_3d += loss_func(keypoints[0], keypoints[1])
                kp_number += coord.shape[2]

            losses["loss_3d"] = loss_3d
            losses["mean_loss_3d"] = mean_3d_loss
            losses["feature_norm"] = norm
            losses["kp number"] = kp_number
        
        loss = loss_cls.clone()
        if "dec" in masks:
            loss_mask = loss_mask.mean()

            if not PRETRAIN:
                loss += cfg.NET.MASK_LOSS_BCE * loss_mask

            assert not "pseudo" in masks
            masks["pseudo"] = pseudo_gt
            losses["loss_mask"] = loss_mask.item()

        # add 3d consistency loss
        if self.dataset in ["wikiscenes_corr", "wikiscenes_corr_semi"] and train:
            # when epoch >= self.pretrain, finetune classification loss
            loss += losses["loss_3d"] * (self.loss_3d / cfg.TRAIN.BATCH_SIZE * 1 if (self.pretrain < 0 or epoch < self.pretrain) else 0)

        losses["loss"] = loss.item()

        if train:
            self.optim_enc.zero_grad()
            loss.backward()
            self.optim_enc.step()

        for mask_key, mask_val in masks.items():
            masks[mask_key] = masks[mask_key].detach()

        mask_logits = mask_logits.detach()

        if visualise:
            self._visualise(epoch, image, masks, mask_logits, cls_out, gt_labels, save_image, info)
        if debug:
            self._visualise_corr(epoch, image, corr, coord_arr, save_image, info)


        # make sure to cut the return values from graph
        return losses, cls_out.detach(), masks, mask_logits

    def train_epoch(self, epoch):
        self.scheduler.step()
        self.enc.train()

        stat = StatManager()
        stat.add_val("loss")
        stat.add_val("loss_cls")
        stat.add_val("loss_fg")
        stat.add_val("loss_bce")

        # adding stats for classes
        timer = Timer("New Epoch: ")
        train_step = partial(self.step, train=True, visualise=False)

        preds_all = list()
        targets_all = list()
        related_all = list()

        for i, (image, related_labels, gt_labels, _) in enumerate(self.trainloader):

            if self.multi_label:
                gt_labels = related_labels

            if self.dataset == "wikiscenes_corr":
                corr = image['corr']
                for j in range(len(corr)):
                    corr[j] = json.loads(corr[j])
                image = torch.cat([image['1'], image['2']], 0)  # [32, 3, 224, 224]
                image_corr = {"image": image, "corr": corr}
                gt_labels = torch.cat(gt_labels, 0)
                related_labels = torch.cat(related_labels, 0)

                losses, cls_out, _, _ = train_step(epoch, image_corr, gt_labels)
            elif self.dataset == "wikiscenes_corr_semi":
                corr = image['corr']
                for j in range(len(corr)):
                    corr[j] = json.loads(corr[j])
                image = torch.cat([image['1'], image['2']], 0)  # [32, 3, 224, 224]
                image_corr = {"image": image, "corr": corr}
                # just place holder...
                gt_labels = torch.cat([gt_labels, gt_labels], 0)
                related_labels = torch.cat([related_labels, related_labels], 0)

                losses, cls_out, _, _ = train_step(epoch, image_corr, gt_labels)

                # discard the place holder
                bs = gt_labels.shape[0] // 2
                cls_out = cls_out[:bs]
                related_labels = related_labels[:bs]
                gt_labels = gt_labels[:bs]
            else:
                losses, cls_out, _, _ = train_step(epoch, image, gt_labels)

            cls_sigmoid = torch.sigmoid(cls_out.cpu()).numpy()
            preds_all.append(cls_sigmoid)
            targets_all.append(gt_labels.cpu().numpy())
            related_all.append(related_labels.cpu().numpy())


            if self.fixed_batch is None or "points" not in self.fixed_batch:
                self.fixed_batch = dict()
                paired_image_1 = torch.cat([image[j:j+1] for j in range(cfg.TRAIN.BATCH_SIZE) if corr[j]], 0)
                paired_image_2 = torch.cat([image[j+cfg.TRAIN.BATCH_SIZE:j+1+cfg.TRAIN.BATCH_SIZE] for j in range(cfg.TRAIN.BATCH_SIZE) if corr[j]], 0)
                paired_image = torch.cat([paired_image_1, paired_image_2], 0)
                paired_gt_labels = gt_labels[:paired_image.shape[0]]
                self.fixed_batch["image"]   = paired_image.clone()
                self.fixed_batch["labels"]  = paired_gt_labels.clone()
                random_points = list()
                for j in range(paired_image.shape[0] // 2):
                    # 3 points per image in a batch
                    random_points.append([{"rx": random.random(), "ry": random.random()} for k in range(3)])
                self.fixed_batch["points"] = random_points
                torch.save(self.fixed_batch, self.fixed_batch_path)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            # intermediate logging
            if i % 10 == 0:
                msg =  "Loss [{:04d}]: ".format(i)
                for loss_key, loss_val in losses.items():
                    msg += "{}: {:.4f} | ".format(loss_key, loss_val)
                
                msg += " | Im/Sec: {:.1f}".format(i * cfg.TRAIN.BATCH_SIZE / timer.get_stage_elapsed())
                print(msg)
                sys.stdout.flush()

            del image, gt_labels

            if DEBUG and i > 0:
                break

        def publish_loss(stats, name, t, prefix='data/'):
            print("{}: {:4.3f}".format(name, stats.summarize_key(name)))
            self.writer.add_scalar(prefix + name, stats.summarize_key(name), t)

        for stat_key in stat.vals.keys():
            publish_loss(stat, stat_key, epoch)

        # plotting learning rate
        for ii, l in enumerate(self.optim_enc.param_groups):
            print("Learning rate [{}]: {:4.3e}".format(ii, l['lr']))
            self.writer.add_scalar('lr/enc_group_%02d' % ii, l['lr'], epoch)



        # self.writer.add_scalar('lr/bg_baseline', self.enc.module.mean.item(), epoch)
        with torch.no_grad():
            # the second parameter is not used
            image_raw = self.denorm(self.fixed_batch["image"].clone())
            self.enc.eval()
            _, _, masks, _, _, _ = self.enc(self.fixed_batch["image"], image_raw, self.fixed_batch["labels"])
            feature = masks["feature"].cpu()
            s, _, w, h = feature.shape
            colormaps = list()
            for i in range(s // 2):
                raw = [image_raw[k] for k in [i, i + s // 2]]
                for j in range(3):
                    x = int(self.fixed_batch["points"][i][j]["rx"] * cfg.DATASET.CROP_SIZE)
                    y = int(self.fixed_batch["points"][i][j]["ry"] * cfg.DATASET.CROP_SIZE)
                    fx = int(self.fixed_batch["points"][i][j]["rx"] * w)
                    fy = int(self.fixed_batch["points"][i][j]["ry"] * h)
                    selected_feature = feature[i][:,fy, fx]
                    heat = [torch.norm((feature[k] - selected_feature[:,None,None]), dim=0) for k in [i, i + s // 2]]

                    # normalize separately
                    min_ = torch.min(heat[0].min(), heat[1].min())
                    range_ = torch.max(heat[0].max(), heat[1].max()) - min_
                    heat = [(heat[k] - min_) / range_ for k in [0,1]]

                    heat[0] = (heat[0] - min_) / range_
                    heat[0] = 1.0 - heat[0]
                    heat[1] = (heat[1] - min_) / range_
                    heat[1] = 1.0 - heat[1]

                    # put color
                    colormap = [apply_colormap_on_image(raw[k], heat[k], 'jet') for k in [0,1]]

                    # draw a cross
                    green = (0, 255, 0)
                    mark_size = 5
                    for k in range(x - mark_size, x + 1 + mark_size):
                        if 0 <= k < colormap[0].size[0]:
                            colormap[0].putpixel((k, y), green)
                    for k in range(y - mark_size, y + 1 + mark_size):
                        if 0 <= k < colormap[0].size[1]:
                            colormap[0].putpixel((x, k), green)
                    colormap = [transforms.ToTensor()(colormap[k]) for k in [0,1]]
                    colormaps.append(colormap)
        self.write_image(colormaps, epoch)
        self.count_acc(targets_all, preds_all, related_all, self.writer, epoch)

        # visualising
        # self.enc.eval()
        # with torch.no_grad():
        #     self.step(epoch, self.fixed_batch["image"], \
        #                      self.fixed_batch["labels"], \
        #                      train=False, visualise=True)

    def export(self, loader):

        if self.export_set == "train" and False:
            self.enc.train()
        else:
            self.enc.eval()

        assert self.dataset in ["wikiscenes_corr", "wikiscenes_corr_export"] , "must provide corr"

        counter = dict()

        preds_all = list()
        targets_all = list()
        related_all = list()
        score = list()

        for i, (image, related_labels, gt_labels, info) in enumerate(loader):
            corr = image['corr']
            path = image['path']
            no_kp = image['no. of kp']
            no_pair = image['no. of pair']
            for j in range(len(corr)):
                corr[j] = json.loads(corr[j])
            # image = image['1']
            if self.export_task == "features":
                image = torch.cat([image['1'], image['2']], 0)
                gt_labels = torch.cat(gt_labels, 0)
                related_labels = torch.cat(related_labels, 0)
            else:
                image = image['1']
                gt_labels = gt_labels[0]
                related_labels = related_labels[0]
            image_corr = {"image": image, "corr": corr}

            # denorm image
            image_raw = self.denorm(image.clone())

            extract_layer = "decoder" if self.export_task == "kp_feature" else "score"

            # classification
            with torch.no_grad():
                cls_out, cls_fg, masks, mask_logits, pseudo_gt, loss_mask = self.enc(image, image_raw, gt_labels, extract_layer)
            masks["pseudo"] = pseudo_gt

            criterion = nn.BCEWithLogitsLoss(reduction='none')
            if self.export_task == "AP":

                loss = criterion(cls_out.cpu(), gt_labels.cpu()).mean(1).numpy()
                for k in range(loss.shape[0]):
                    labels = [self.classNames[j+1] for j in range(self.nclass-1) if gt_labels[k][j] > 0]
                    score.append([path[k], no_kp[k].item(), no_pair[k].item(), loss[k], str(labels)])

            elif self.export_task in ["kp_score", "kp_feature"]:

                feature = masks["feature"].detach()

                b, c, h, w = feature.shape
                for k in range(b):
                    if not corr[k]:
                        continue
                    # k * 4
                    names, p = corr[k].keys(), corr[k].values()
                    coord = torch.tensor(list(p))
                    if cfg.NUM_GPUS != 0:
                        coord = coord.cuda()
                    # reshape, permute to allow grid_sample, [N, Hout, Wout, 2]
                    coord = coord.reshape(1, -1, 2, 2).permute(2, 0, 1, 3)[:1].contiguous()
                    coord = coord * 2 - 1

                    keypoints = F.grid_sample(feature[k:k+1], torch.flip(coord, [-1])).detach().cpu().numpy()
                    for j, name in enumerate(names):
                        if name not in counter:
                            counter[name] = list()
                        counter[name].append(keypoints[0,:,0,j])

            elif self.export_task == "segmentation":
                for mask_key, mask_val in masks.items():
                    masks[mask_key] = masks[mask_key].detach()
                mask_logits = mask_logits.detach()
                self._visualise(0, image, masks, mask_logits, cls_out, gt_labels+related_labels, True, info[0])
                if i > 100:
                    break

            elif self.export_task == "acc":
                cls_sigmoid = torch.sigmoid(cls_out.cpu()).numpy()
                preds_all.append(cls_sigmoid)
                targets_all.append(gt_labels.cpu().numpy())
                related_all.append(related_labels.cpu().numpy())

            elif self.export_task == "features":
                feature = masks["feature"].detach()
                s, _, w, h = feature.shape #b, c, h, w
                colormaps = list()
                for m in range(s // 2):
                    raw = [image_raw[k] for k in [m, m + s // 2]]
                    if not corr[m]:
                        continue
                    # k * 4
                    sorted_corr = dict(sorted(corr[m].items()))
                    names, p = sorted_corr.keys(), sorted_corr.values()
                    coord = torch.tensor(list(p))
                    if cfg.NUM_GPUS != 0:
                        coord = coord.cuda()
                    for j in range(min(coord.shape[0],3)):
                        x = int(coord[j,1] * cfg.DATASET.CROP_SIZE)
                        y = int(coord[j,0] * cfg.DATASET.CROP_SIZE)
                        fx = int(coord[j,1] * w)
                        fy = int(coord[j,0] * h)
                        selected_feature = feature[m][:, fy, fx]
                        heat = [torch.norm((feature[k] - selected_feature[:, None, None]), dim=0) for k in
                                [m, m + s // 2]]

                        # normalize separately
                        min_ = torch.min(heat[0].min(), heat[1].min())
                        range_ = torch.max(heat[0].max(), heat[1].max()) - min_

                        heat[0] = (heat[0] - min_) / range_
                        heat[0] = 1.0 - heat[0]
                        heat[1] = (heat[1] - min_) / range_
                        heat[1] = 1.0 - heat[1]

                        # put color
                        colormap = [apply_colormap_on_image(raw[k], heat[k].cpu(), 'jet') for k in [0, 1]]
                        path_save = "./logs/vis_features"
                        overlay_map_on_im(torch.cat((raw[0],raw[1]), dim = 2), torch.cat((heat[0],heat[1]), dim=1).cpu(),(x,y),
                                          "{}/new_{:0>4}.{:0>4}.{:0>4}.jpg".format(path_save, i, m, j))
                        # draw a cross
                        green = (250, 250, 250)
                        mark_size = 8
                        for k in range(x - mark_size, x + 1 + mark_size):
                            if 0 <= k < colormap[0].size[0]:
                                colormap[0].putpixel((k, y), green)
                                colormap[0].putpixel((k, y-1), green)
                                colormap[0].putpixel((k, y-2), green)
                                colormap[0].putpixel((k, y+1), green)
                                colormap[0].putpixel((k, y+2), green)
                        for k in range(y - mark_size, y + 1 + mark_size):
                            if 0 <= k < colormap[0].size[1]:
                                colormap[0].putpixel((x, k), green)
                                colormap[0].putpixel((x+1, k), green)
                                colormap[0].putpixel((x+2, k), green)
                                colormap[0].putpixel((x-1, k), green)
                                colormap[0].putpixel((x-2, k), green)
                        colormap = [transforms.ToTensor()(colormap[k]) for k in [0, 1]]
                        colormaps.append(torch.cat([colormap[0], colormap[1]], dim=2))
                print('saving feature visualizations...')
                path_save = "./logs/vis_features"
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
                for j, image in enumerate(colormaps):
                    plt.imshow(image.permute(1, 2, 0))
                    plt.axis('off')
                    plt.savefig("{}/{:0>4}.{:0>4}.jpg".format(path_save, i, j))

        if self.export_task in ["kp_score", "kp_feature"]:
            save_path = "./logs/{}/landmark{}/".format(self.export_task, self.export_set)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open("./logs/{}/landmark{}/export_{}_{}_{}_{}.pkl".format(self.export_task, self.export_set, self.export_task, self.export_set, self.run, args.resume), 'wb') as f:
                cPickle.dump(counter, f)
        elif self.export_task == "AP":
            # rank ascending by loss
            score = sorted(score, key=lambda t: t[3])
            header = ["path", "# keypoint", "# pairs", "loss", "labels"]
            df = pandas.DataFrame(score, columns=header)
            df.to_excel("./logs/export_{}_{}_{}_{}.xlsx".format(self.export_task, self.export_set, self.run, args.resume), index=False)
        elif self.export_task == "acc":
            self.count_acc(targets_all, preds_all, related_all, None, 0)
        return



    def _mask_rgb(self, masks, image_norm):
        # visualising masks
        masks_conf, masks_idx = torch.max(masks, 1)
        masks_conf = masks_conf - F.relu(masks_conf - 1, 0)

        masks_idx_rgb = self._apply_cmap(masks_idx.cpu(), masks_conf.cpu())
        return 0.3 * image_norm + 0.7 * masks_idx_rgb

    def _init_norm(self):
        self.trainloader.dataset.set_norm(self.enc.normalize)
        if self.export_task == "None" or self.export_set == "val_s":
            self.valloader.dataset.set_norm(self.enc.normalize)
        # if self.export_task == "None" or self.export_set == "val_l":
        #     self.valloader_single.dataset.set_norm(self.enc.normalize)
        if self.export_task == "None":
            self.valloader_easy.dataset.set_norm(self.enc.normalize)
        self.trainloader_val.dataset.set_norm(self.enc.normalize)

    def _apply_cmap(self, mask_idx, mask_conf):
        palette = self.trainloader.dataset.get_palette()

        masks = []
        col = Colorize()
        mask_conf = mask_conf.float() / 255.0
        for mask, conf in zip(mask_idx.split(1), mask_conf.split(1)):
            m = col(mask).float()
            m = m * conf
            masks.append(m[None, ...])

        return torch.cat(masks, 0)

    def validation(self, epoch, writer, loader, checkpoint=False):

        stat = StatManager()

        # Fast test during the training
        def eval_batch(image, gt_labels, info):

            # do not save the images to save time
            losses, cls, masks, mask_logits = \
                    self.step(epoch, image, gt_labels, train=False, visualise=False, save_image=True, info=info)

            for loss_key, loss_val in losses.items():
                stat.update_stats(loss_key, loss_val)

            return cls.cpu(), masks, mask_logits.cpu()

        self.enc.eval()

        # class ground truth
        targets_all = []

        # class predictions
        preds_all = []
        related_all = list()

        def add_stats(means, stds, x):
            means.append(x.mean())
            stds.append(x.std())

        for n, (image, related_labels, gt_labels, info) in enumerate(loader):

            if self.dataset == "wikiscenes_corr":
                info = info[0]
                corr = image['corr']
                for i in range(len(corr)):
                    corr[i] = json.loads(corr[i])
                # image = torch.cat([image['1'], image['2']], 0)
                # not validate the random selected ones
                image = image['1']
                image_corr = {"image": image, "corr": corr}
                gt_labels = gt_labels[0]
                related_labels = related_labels[0]
            elif self.dataset == "wikiscenes_corr_semi":
                corr = image['corr']
                for i in range(len(corr)):
                    corr[i] = json.loads(corr[i])
                # not validate the random selected ones
                image = image['1']
                image_corr = {"image": image, "corr": corr}

            with torch.no_grad():
                cls_raw, masks_all, mask_logits = eval_batch(
                    image_corr if self.dataset in ["wikiscenes_corr", "wikiscenes_corr_semi"] else image,
                    gt_labels, info)

            cls_sigmoid = torch.sigmoid(cls_raw).numpy()

            preds_all.append(cls_sigmoid)
            targets_all.append(gt_labels.cpu().numpy())
            related_all.append(related_labels.cpu().numpy())

        self.count_acc(targets_all, preds_all, related_all, writer, epoch)

        # total classification loss
        for stat_key in stat.vals.keys():
            writer.add_scalar('all/{}'.format(stat_key), stat.summarize_key(stat_key), epoch)

        # if checkpoint and epoch >= cfg.TRAIN.PRETRAIN: 
        if checkpoint: 
            # we will use mAP - mask_loss as our proxy score
            # to save the best checkpoint so far
            proxy_score = 1 - stat.summarize_key("loss")
            writer.add_scalar('all/checkpoint_score', proxy_score, epoch)
            self.checkpoint_best(proxy_score, epoch)

    def count_acc(self, targets_all, preds_all, related_gt, writer, epoch):

        #
        # classification
        #
        targets_stacked = np.vstack(targets_all)
        preds_stacked = np.vstack(preds_all)
        related_stacked = np.vstack(related_gt)
        aps = average_precision_score(related_stacked if self.multi_label else targets_stacked, preds_stacked, average=None)

        if not self.multi_label:
            y_true = targets_stacked.argmax(1)
            y_pred = preds_stacked.argmax(1)
            acc = accuracy_score(y_true, y_pred)

            # per class accuracy
            cm = confusion_matrix(y_true, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_d = cm.diagonal()

            plt.figure()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.pink_r)
            plt.title("confusion matrix")
            plt.colorbar()
            tick_marks = np.arange(self.nclass - 1)
            plt.xticks(tick_marks, self.classNames[1:-1], rotation=45)
            plt.yticks(tick_marks, self.classNames[1:-1])
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            matrix = Image.open(buf)
            matrix = transforms.ToTensor()(matrix)

        # 'easy' accuracy
        mcm = multilabel_confusion_matrix(related_stacked, (preds_stacked.max(axis=1, keepdims=1)==preds_stacked).astype(np.float))
        easy_acc = mcm[:,1,1].sum() / preds_stacked.shape[0]

        # skip BG AP
        offset = self.nclass - aps.size
        assert offset == 1, 'Class number mismatch'

        classNames = self.classNames[offset:-1]
        for ni, className in enumerate(classNames):
            if writer is not None:
                writer.add_scalar('%02d_%s/AP' % (ni + offset, className), aps[ni], epoch)
                if not self.multi_label:
                    writer.add_scalar('%02d_%s/acc' % (ni + offset, className), cm_d[ni], epoch)
            # ap_ = "AP_{}: {:4.3f}".format(className, aps[ni])
            # if not multi_label:
            #     acc_ = ", acc_{}: {:4.3f}".format(className, cm_d[ni])
            # print(ap_ + acc_ if not multi_label else "")

        meanAP = np.mean(aps)
        if writer is not None:
            writer.add_scalar('all_wo_BG/mAP', meanAP, epoch)
            writer.add_scalar('all_wo_BG/easy_acc', easy_acc, epoch)
            if not self.multi_label:
                writer.add_scalar('all_wo_BG/acc', acc, epoch)
                writer.add_scalar('all_wo_BG/acc(averaged by classes)', cm_d.mean(), epoch)
                writer.add_image('confusion_matrix', matrix, epoch)
        # print('mAP: {:4.3f}, acc: {:4.3f}'.format(meanAP, acc))

    def _visualise_corr(self, epoch, image, corr, coord_arr=None, for_save=False, info=None):
        image_norm = self.denorm(image.clone()).cpu()
        visual = [image_norm]
        # ready to assemble
        visual_logits = torch.cat(visual, -1)  #[16, 3, 224, 224]
        self._visualise_corr_grid(visual_logits, corr, coord_arr, epoch)
        if for_save:
            self.visual_times += 1

    def _visualise(self, epoch, image, masks, mask_logits, cls_out, gt_labels, for_save=False, info=None):
        image_norm = self.denorm(image.clone()).cpu()
        visual = [image_norm]

        if "cam" in masks:
            visual.append(self._mask_rgb(masks["cam"], image_norm))

        if "dec" in masks:
            visual.append(self._mask_rgb(masks["dec"], image_norm))

        if "pseudo" in masks:
            pseudo_gt_rgb = self._mask_rgb(masks["pseudo"], image_norm)

            # cancel ambiguous
            ambiguous = 1 - masks["pseudo"].sum(1, keepdim=True).cpu()
            pseudo_gt_rgb = ambiguous * image_norm + (1 - ambiguous) * pseudo_gt_rgb
            visual.append(pseudo_gt_rgb)

        if "full" in masks:
            bg = masks["full"][:,0,:,:]
            for i in range(1, self.nclass):
                temp = torch.zeros_like(masks["full"])
                temp[:,0,:,:] = bg
                temp[:,i,:,:] = masks["full"][:,i,:,:]
                visual.append(self._mask_rgb(temp, image_norm))

        # ready to assemble
        visual_logits = torch.cat(visual, -1)
        self._visualise_grid(visual_logits, gt_labels, epoch, scores=cls_out, save_image=for_save, epoch=epoch, index=self.visual_times, info=info)
        if for_save:
            self.visual_times += 1

if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])

    # Reading the config
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Config: \n", cfg)

    trainer = DecTrainer(args)

    timer = Timer()
    def time_call(func, msg, *args, **kwargs):
        timer.reset_stage()
        func(*args, **kwargs)
        print(msg + (" {:3.2}m".format(timer.get_stage_elapsed() / 60.)))

    if args.export != "None":
        loader = {
            "train": trainer.trainloader,
            "val": trainer.valloader,
            "val_easy": trainer.valloader_easy
        }
        if args.export_set in loader:
            trainer.export(loader[args.export_set])
        else:
            assert args.export in ["kp_score", "kp_feature"], "wrong export task or dataset"
            trainer.export(loader["train"])
        quit()

    for epoch in range(trainer.start_epoch, cfg.TRAIN.NUM_EPOCHS + 1):
        print("Epoch >>> ", epoch)

        log_int = 5 if DEBUG else 2
        if epoch % log_int == 0:
            with torch.no_grad():
                time_call(trainer.validation, "Validation / Val(easy): ", epoch, trainer.writer_val_easy, trainer.valloader_easy, checkpoint=False)
                time_call(trainer.validation, "Validation / Val: ", epoch, trainer.writer_val, trainer.valloader, checkpoint=True)

        time_call(trainer.train_epoch, "Train epoch: ", epoch)