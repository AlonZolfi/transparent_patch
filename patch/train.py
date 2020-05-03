import sys
import os
if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/transparent_patch/patch')
    sys.path.append('/home/zolfi/transparent_patch/pytorch-yolo2')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# else:
# from torchviz import make_dot
#     import cv2

import torch
from torch import nn
from torch.optim import Adam

from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path

from config import patch_config_types
from darknet import Darknet
from load_data import SplitDataset
from nn_modules import MaxProbExtractor, PatchApplier, PatchTrainer, TotalVariation, IoU, PreserveDetections
from patch_utils import EarlyStopping

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

        self.patch_applier = PatchApplier()

        # self.prob_extractor = MaxProbExtractor(
        #     weight=self.config.max_prob_weight,
        #     cls_id=self.config.class_id,
        #     num_cls=self.config.num_classes,
        #     config=self.config,
        #     num_anchor=self.yolo.num_anchors)

        self.preserve_detec = PreserveDetections(
            weight_cls=self.config.max_prob_weight,
            weight_others=self.config.pres_det_weight,
            cls_id=self.config.class_id,
            num_cls=self.config.num_classes,
            config=self.config,
            num_anchor=self.yolo.num_anchors
        )
        self.total_variation = TotalVariation(weight=self.config.tv_weight)
        # self.iou = IoU(weight=self.config.iou_weight)
        # self.patch_transformer = PatchTransformer()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)

        self.max_tv = torch.tensor(self.config.max_tv).to(device)

        self.set_multiple_gpu()
        self.set_to_device()

        if 'SLURM_JOBID' not in os.environ.keys():
            self.current_dir = "saved_patch/" + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S")
        else:
            self.current_dir = "saved_patch/" + time.strftime("%d-%m-%Y") + '_' + os.environ['SLURM_JOBID']
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)

        torch.cuda.manual_seed(SEED)

        # self.writer = self.init_tensorboard(mode)

    def set_to_device(self):
        self.yolo.to(device)
        self.patch_applier.to(device)
        # self.prob_extractor.to(device)
        self.total_variation.to(device)
        self.preserve_detec.to(device)
        # self.iou.to(device)
        # self.patch_transformer.to(device)
        # self.nps_calculator.to(device)

    def set_multiple_gpu(self):
        if torch.cuda.device_count() > 1:
            print("more than 1")
            self.yolo = torch.nn.DataParallel(self.yolo)
            self.patch_applier = torch.nn.DataParallel(self.patch_applier)
            # self.prob_extractor = torch.nn.DataParallel(self.prob_extractor)
            self.total_variation = torch.nn.DataParallel(self.total_variation)
            self.preserve_detec = torch.nn.DataParallel(self.preserve_detec)
            # self.iou = torch.nn.DataParallel(self.iou)
            # self.patch_transformer = torch.nn.DataParallel(self.patch_transformer)
            # self.nps_calculator = torch.nn.DataParallel(self.nps_calculator)
            torch.cuda.manual_seed_all(SEED)

    def train(self):
        img_size = self.get_image_size()
        max_lab = 39

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

        alpha_tensor_cpu = torch.full((1, img_size, img_size), dtype=torch.float32, fill_value=0.1)
        alpha_tensor_cpu.requires_grad_(True)
        adv_patch_cpu = torch.rand((3, img_size, img_size), dtype=torch.float32)
        adv_patch_cpu.requires_grad_(True)

        epoch_length = len(train_loader)
        print(f'One epoch is {epoch_length} batches')

        # optimizer = torch.optim.Adam(self.patch_trainer.parameters(),
        optimizer = Adam([adv_patch_cpu, alpha_tensor_cpu],
                         lr=self.config.start_learning_rate,
                         amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(delta=1e-4, current_dir=self.current_dir)

        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            max_prob_loss = 0.0
            tv_loss = 0.0
            cor_det_loss = 0.0
            train_loss = 0.0
            val_loss = 0.0
            adv_patch = adv_patch_cpu.to(device)
            alpha_tensor = alpha_tensor_cpu.to(device)

            progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, maxprob-loss: {:.6}, tv-loss: {:.6}, corr det-loss: {:.6}'

            for i_batch, (img_batch, lab_batch) in progress_bar:
                # move tensors to cuda
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)

                # forward prop
                # adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, True, False)
                # adv_patch_cpu = self.patch_trainer(adv_patch_cpu1)  # update patch
                # adv_patch = adv_patch_cpu.to(device)

                p_img_batch = self.patch_applier(img_batch, adv_patch, alpha_tensor)  # apply patch on a batch of images

                output_patch = self.yolo(p_img_batch)  # get yolo output
                output_clean = self.yolo(img_batch)  # get yolo output
                # max_prob = self.prob_extractor(output_patch)  # extract probabilities for our class
                max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, output_clean)
                tv = self.total_variation(adv_patch)  # calculate patch total variation

                # iou = self.iou()
                # nps = self.nps_calculator(adv_patch)

                # calculate loss
                loss, loss_arr = self.loss_function(max_prob, tv, cor_det)

                # save losses
                max_prob_loss += loss_arr[0].item()
                tv_loss += loss_arr[1].item()
                cor_det_loss += loss_arr[2].item()
                train_loss += loss.item()


                # back prop
                optimizer.zero_grad()
                loss.backward()

                # update parameters
                optimizer.step()

                # adv_patch_cpu1 = torch.clamp(adv_patch_cpu1, 0, 1)

                # clear gpu
                # del img_batch, lab_batch, loss
                # torch.cuda.empty_cache()
                # make_dot(loss).render("prop_path/iter"+str(i_batch), format="png")
                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss/(i_batch+1),
                                                                  max_prob_loss/(i_batch+1),
                                                                  tv_loss/(i_batch+1),
                                                                  cor_det_loss/(i_batch+1)))

                if i_batch + 1 == epoch_length:
                    # calculate epoch losses
                    train_loss = train_loss / epoch_length
                    train_losses.append(train_loss)

                    max_prob_loss /= epoch_length
                    tv_loss /= epoch_length
                    cor_det_loss /= epoch_length

                    # check on validation
                    val_loss = self.evaluate_loss(val_loader, adv_patch, alpha_tensor)
                    val_losses.append(val_loss)

                    prog_bar_desc += ', val-loss: {:.6}'
                    progress_bar.set_postfix_str(prog_bar_desc.format(train_loss,
                                                                      max_prob_loss,
                                                                      tv_loss,
                                                                      cor_det_loss,
                                                                      val_loss))

            # check if loss has decreased
            early_stop(val_loss, adv_patch, epoch)
            if early_stop.early_stop:
                print("Training stopped - early stopping")
                break

            scheduler.step(train_loss)

        # save patch with alpha layer
        final_patch = torch.cat([adv_patch_cpu.to(device), alpha_tensor_cpu.to(device)])
        transforms.ToPILImage()(final_patch.cpu()).save(self.current_dir+'/final_patch.png', 'PNG')

        # plot train and val loss
        self.plot_loss(train_losses, val_losses)

        # calculate test loss
        test_loss = self.evaluate_loss(test_loader, adv_patch_cpu, alpha_tensor_cpu)
        print("Test loss: " + str(test_loss))
        print("Training finished")

    def get_image_size(self):
        if type(self.yolo) == nn.DataParallel:
            img_size = self.yolo.module.height
        else:
            img_size = self.yolo.height
        return int(img_size)

    def evaluate_loss(self, loader, adv_patch, alpha):
        total_loss = 0.0
        for img_batch, lab_batch in loader:
            with torch.no_grad():
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                p_img_batch = self.patch_applier(img_batch, adv_patch, alpha)
                output_patch = self.yolo(p_img_batch)
                output_clean = self.yolo(img_batch)
                # max_prob = self.prob_extractor(output_patch)
                tv = self.total_variation(adv_patch)
                max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, output_clean)
                batch_loss, _ = self.loss_function(max_prob, tv, cor_det)
                total_loss += batch_loss.item()
                # del img_batch, v_loss
        return total_loss / len(loader)

    def plot_loss(self, train_loss, val_loss):
        epochs = [x+1 for x in range(len(train_loss))]
        plt.plot(epochs, train_loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir+'/loss_plt.png')

    def loss_function(self, max_prob, tv, cor_det):
        max_prob_loss = torch.mean(max_prob)
        tv_loss = torch.max(tv, self.max_tv)
        cor_det_loss = torch.mean(cor_det)
        return max_prob_loss + tv_loss + cor_det_loss, [max_prob_loss, tv_loss, cor_det_loss]


def main():
    patch_train = TrainPatch('private')
    patch_train.train()


if __name__ == '__main__':
    main()
