import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd
import torch.nn.functional as F

from torchvision import transforms
from torchviz import make_dot

import numpy as np
from tqdm import tqdm
import cv2

from config import patch_config_types
from darknet import Darknet
from load_data import LisaDataset
from nn_modules import MaxProbExtractor, PatchApplier, PatchTrainer, TotalVariation, IoU
from PIL import Image, ImageDraw

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainPatch:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()

        self.yolo = Darknet(self.config.cfg_file)
        self.yolo.load_weights(self.config.weight_file)
        self.yolo.eval()

        self.patch_trainer = PatchTrainer(
            self.config.num_of_dots,
            self.get_image_size())

        self.patch_applier = PatchApplier(self.config.alpha)

        self.prob_extractor = MaxProbExtractor(
            weight=self.config.tv_weight,
            cls_id=self.config.class_id,
            num_cls=self.config.num_classes,
            config=self.config,
            num_anchor=self.yolo.num_anchors)

        self.total_variation = TotalVariation(weight=self.config.max_prob_weight)
        self.iou = IoU(weight=self.config.max_prob_weight)
        # self.patch_transformer = PatchTransformer()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)

        self.set_multiple_gpu()
        self.set_to_device()

        # self.writer = self.init_tensorboard(mode)

    def set_to_device(self):
        self.yolo.to(device)
        self.patch_applier.to(device)
        self.prob_extractor.to(device)
        self.total_variation.to(device)
        self.iou.to(device)
        # self.patch_transformer.to(device)
        # self.nps_calculator.to(device)

    def set_multiple_gpu(self):
        if torch.cuda.device_count() > 1:
            print("more than 1")
            self.yolo = torch.nn.DataParallel(self.yolo)
            self.patch_applier = torch.nn.DataParallel(self.patch_applier)
            self.prob_extractor = torch.nn.DataParallel(self.prob_extractor)
            self.total_variation = torch.nn.DataParallel(self.total_variation)
            self.iou = torch.nn.DataParallel(self.iou)
            # self.patch_transformer = torch.nn.DataParallel(self.patch_transformer)
            # self.nps_calculator = torch.nn.DataParallel(self.nps_calculator)

    def train(self):
        img_size = self.get_image_size()
        max_lab = 14

        image_loader = DataLoader(dataset=LisaDataset(img_dir=self.config.img_dir,
                                                      lab_dir=self.config.lab_dir,
                                                      max_lab=max_lab,
                                                      img_size=img_size,
                                                      transform=transforms.ToTensor),
                                  batch_size=self.config.batch_size,
                                  shuffle=True,
                                  num_workers=4)

        # adv_patch_cpu1 = torch.full((3, img_size, img_size), dtype=torch.float32, fill_value=1)
        adv_patch_cpu1 = torch.rand((3, img_size, img_size), dtype=torch.float32)
        adv_patch_cpu1.requires_grad_(True)

        epoch_length = len(image_loader)
        print(f'One epoch is {epoch_length} batches')

        # optimizer = torch.optim.Adam(self.patch_trainer.parameters(),
        optimizer = torch.optim.Adam([adv_patch_cpu1],
                                     lr=self.config.start_learning_rate,
                                     amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        for epoch in range(self.config.epochs):
            epoch_total_loss = 0
            epoch_max_prob_loss = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(image_loader),
                                                        desc=f'Running epoch {epoch}',
                                                        total=epoch_length):
                with autograd.detect_anomaly():
                    # move tensors to cuda
                    img_batch = img_batch.to(device)
                    lab_batch = lab_batch.to(device)

                    # forward prop
                    # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, True, False)
                    # adv_patch_cpu = self.patch_trainer(adv_patch_cpu1)  # update patch
                    # adv_patch = adv_patch_cpu.to(device)
                    adv_patch = adv_patch_cpu1.to(device)

                    p_img_batch = self.patch_applier(img_batch, adv_patch)  # apply patch on a batch of images

                    output = self.yolo(p_img_batch)  # get yolo output

                    max_prob = self.prob_extractor(output)  # extract probabilities for our class

                    tv = self.total_variation(adv_patch)  # calculate patch total variation

                    # iou = self.iou()
                    # nps = self.nps_calculator(adv_patch)

                    # calculate loss
                    max_prob_loss = torch.mean(max_prob)
                    tv_loss = torch.max(tv, torch.tensor(self.config.max_tv).to(device))
                    loss = max_prob_loss + tv_loss

                    # save losses
                    epoch_max_prob_loss += max_prob_loss
                    epoch_total_loss += loss

                    # back prop
                    optimizer.zero_grad()
                    loss.backward()

                    # update parameters
                    optimizer.step()

                    # make_dot(loss).render("prop_path/iter"+str(i_batch), format="png")
            transforms.ToPILImage('RGB')(adv_patch.cpu()).show()
            scheduler.step(epoch_total_loss/epoch_length)

    def get_image_size(self):
        if type(self.yolo) == nn.DataParallel:
            img_size = self.yolo.module.height
        else:
            img_size = self.yolo.height
        return int(img_size)


def main():
    patch_train = TrainPatch('base')
    patch_train.train()


if __name__ == '__main__':
    main()
