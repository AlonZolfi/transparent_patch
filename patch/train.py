import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd
import torch.nn.functional as F

from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import patch_config_types
from darknet import Darknet
from load_data import LisaDataset
from nn_modules import MaxProbExtractor, PatchApplier

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainPatch:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()

        self.yolo = Darknet(self.config.cfg_file)
        self.yolo.load_weights(self.config.weight_file)
        self.yolo.eval()

        self.patch_applier = PatchApplier()

        # self.patch_transformer = PatchTransformer()

        self.prob_extractor = MaxProbExtractor(self.config.class_id, self.config.num_classes, self.config)

        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)

        # self.total_variation = TotalVariation()

        self.set_multiple_gpus()
        self.set_to_device()

        # self.writer = self.init_tensorboard(mode)

    def set_to_device(self):
        self.yolo.to(device)
        self.patch_applier.to(device)
        # self.patch_transformer.to(device)
        self.prob_extractor.to(device)
        # self.nps_calculator.to(device)
        # self.total_variation.to(device)

    def set_multiple_gpus(self):
        if torch.cuda.device_count() > 1:
            print("more than 1")
            self.yolo = torch.nn.DataParallel(self.yolo)
            self.patch_applier = torch.nn.DataParallel(self.patch_applier)
            # self.patch_transformer = torch.nn.DataParallel(self.patch_transformer)
            self.prob_extractor = torch.nn.DataParallel(self.prob_extractor)
            # self.nps_calculator = torch.nn.DataParallel(self.nps_calculator)
            # self.total_variation = torch.nn.DataParallel(self.total_variation)

    def train(self):
        img_size = self.get_image_size()
        max_lab = 14

        adv_patch_cpu = self.generate_patch().requires_grad_(True)

        image_loader = DataLoader(LisaDataset(img_dir=self.config.img_dir,
                                              lab_dir=self.config.lab_dir,
                                              max_lab=max_lab,
                                              img_size=img_size,
                                              transform=transforms.ToTensor),
                                  batch_size=3,
                                  shuffle=True,
                                  num_workers=4)

        epoch_length = len(image_loader)
        print(f'One epoch is {epoch_length}')

        optimizer = torch.optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        for epoch in range(self.config.epochs):
            ep_loss = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(image_loader),
                                                        desc=f'Running epoch {epoch}',
                                                        total=epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.to(device)
                    lab_batch = lab_batch.to(device)
                    adv_patch = adv_patch_cpu.to(device)

                    # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_patch)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    output = self.yolo(p_img_batch)
                    max_prob = self.prob_extractor(output)
                    # nps = self.nps_calculator(adv_patch)
                    # tv = self.total_variation(adv_patch)

                    max_prob_coef = 0.1
                    loss = max_prob_coef * max_prob
                    ep_loss += loss

                    loss.backward()
                    optimizer.zero_grad()
                    optimizer.step()

                    adv_patch.clamp_(0, 1)

            scheduler.step(ep_loss/epoch_length)

    def get_image_size(self):
        if type(self.yolo) == nn.DataParallel:
            img_size = self.yolo.module.height
        else:
            img_size = self.yolo.net_info['height']
        return img_size

    def generate_patch(self):
        adv_patch_cpu = torch.ones((4, self.config.patch_size, self.config.patch_size))
        adv_patch_cpu[3].fill_(1)
        return adv_patch_cpu

    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask


def main():
    patch_train = TrainPatch('base')
    patch_train.train()


if __name__ == '__main__':
    main()
