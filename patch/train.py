# import sys
# if sys.base_prefix.__contains__('home/zolfi'):
#     sys.path.append('/home/zolfi/transparent_patch/patch')
#     sys.path.append('/home/zolfi/transparent_patch/pytorch-yolo2')
# else:
#     from torchviz import make_dot
#     import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import autograd

from torchvision import transforms

from tqdm import tqdm

from config import patch_config_types
from darknet import Darknet
from load_data import LisaDataset, SplitDataset
from nn_modules import MaxProbExtractor, PatchApplier, PatchTrainer, TotalVariation, IoU
from patch_utils import EarlyStopping
from PIL import Image, ImageDraw

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 42


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

        torch.cuda.manual_seed(SEED)

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
            torch.cuda.manual_seed_all(SEED)

    def train(self):
        img_size = self.get_image_size()
        max_lab = 14

        split_dataset = SplitDataset(img_dir=self.config.img_dir,
                                     lab_dir=self.config.lab_dir,
                                     max_lab=max_lab,
                                     img_size=img_size,
                                     transform=transforms.Compose([transforms.Resize((img_size, img_size)),
                                                                   transforms.ToTensor()]))
        train_loader, val_loader, test_loader = split_dataset(val_split=0.2,
                                                              test_split=0.2,
                                                              shuffle_dataset=True,
                                                              random_seed=SEED,
                                                              batch_size=self.config.batch_size)

        # adv_patch_cpu1 = torch.full((3, img_size, img_size), dtype=torch.float32, fill_value=1)
        adv_patch_cpu1 = torch.rand((3, img_size, img_size), dtype=torch.float32)
        adv_patch_cpu1.requires_grad_(True)

        epoch_length = len(train_loader)
        print(f'One epoch is {epoch_length} batches')

        # optimizer = torch.optim.Adam(self.patch_trainer.parameters(),
        optimizer = torch.optim.Adam([adv_patch_cpu1],
                                     lr=self.config.start_learning_rate,
                                     amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(delta=1e-4)

        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            train_loss = 0
            adv_patch = adv_patch_cpu1.to(device)

            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader),
                                                        desc=f'Running epoch {epoch}',
                                                        total=epoch_length):
                # move tensors to cuda
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)

                # forward prop
                # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, True, False)
                # adv_patch_cpu = self.patch_trainer(adv_patch_cpu1)  # update patch
                # adv_patch = adv_patch_cpu.to(device)

                p_img_batch = self.patch_applier(img_batch, adv_patch)  # apply patch on a batch of images

                output = self.yolo(p_img_batch)  # get yolo output

                max_prob = self.prob_extractor(output)  # extract probabilities for our class
                tv = self.total_variation(adv_patch)  # calculate patch total variation

                # iou = self.iou()
                # nps = self.nps_calculator(adv_patch)

                # calculate loss
                loss = self.loss_function(max_prob, tv)

                # save losses
                train_loss += loss.item()

                # back prop
                optimizer.zero_grad()
                loss.backward()

                # update parameters
                optimizer.step()

                adv_patch_cpu1 = torch.clamp(adv_patch_cpu1, 0, 1)

                # clear gpu
                del img_batch, lab_batch, loss
                torch.cuda.empty_cache()
                # make_dot(loss).render("prop_path/iter"+str(i_batch), format="png")

            # calculate epoch losses
            train_loss = train_loss / epoch_length
            train_losses.append(train_loss)
            tqdm.write('train loss: ' + str(train_loss), end="")

            # check on validation
            val_loss = self.calc_val_loss(val_loader, adv_patch.data)
            val_losses.append(val_loss)
            print('val loss: ' + str(val_loss))

            # check if loss has decreased
            early_stop(val_loss, adv_patch)
            if early_stop.early_stop:
                print("Training stopped - early stopping")
                break

            scheduler.step(train_loss)

    def get_image_size(self):
        if type(self.yolo) == nn.DataParallel:
            img_size = self.yolo.module.height
        else:
            img_size = self.yolo.height
        return int(img_size)

    def calc_val_loss(self, val_loader, adv_patch):
        val_loss = 0
        for img_batch, lab_batch in val_loader:
            img_batch = img_batch.to(device)
            # lab_batch = lab_batch.to(device)
            p_img_batch = self.patch_applier(img_batch, adv_patch)
            output = self.yolo(p_img_batch)
            max_prob = self.prob_extractor(output)
            tv = self.total_variation(adv_patch)
            v_loss = self.loss_function(max_prob, tv)
            val_loss += v_loss.item()
            del img_batch, v_loss
        return val_loss / len(val_loader)

    def loss_function(self, max_prob, tv):
        max_prob_loss = torch.mean(max_prob)
        tv_loss = torch.max(tv, torch.tensor(self.config.max_tv).to(device))
        return max_prob_loss + tv_loss


def main():
    patch_train = TrainPatch('private')
    patch_train.train()


if __name__ == '__main__':
    main()
