import sys
import os

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/transparent_patch/patch')
    sys.path.append('/home/zolfi/transparent_patch/yolov5-ultralytics')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.optim import Adam
from torchvision import transforms

import numpy as np
import torch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path
import pickle
import datetime
import cv2
import subprocess
from tensorboardX import SummaryWriter

from config import patch_config_types
from models.experimental import attempt_load

# from darknet import Darknet
from load_data import SplitDataset, SplitDataset1
from nn_modules import PatchApplier, DotApplier, WeightClipper, NonPrintabilityScore, DetectionsYolov5, NoiseAmount
from patch_utils import EarlyStopping
from evaluate import EvaluateYOLO


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


seed = 42
set_random_seed(seed)

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainPatch:
    def __init__(self, mode):
        self.config = patch_config_types[mode]()

        self.yolov5 = attempt_load(self.config.weight_file, map_location=device)

        self.img_size = self.config.patch_size

        self.dot_applier = DotApplier(
            self.config.num_of_dots,
            self.img_size,
            self.config.alpha_max,
            self.config.beta_dropoff)

        self.patch_applier = PatchApplier()

        self.non_printability_score = NonPrintabilityScore(
            self.config.print_file,
            self.config.num_of_dots)

        self.eval_type = self.config.eval_data_type
        self.clean_img_dict = np.load('confidences/yolov5/medium/clean_img_conf_lisa_ordered.npy', allow_pickle=True).item()

        self.detections = DetectionsYolov5(
            cls_id=self.config.class_id,
            num_cls=self.config.num_classes,
            config=self.config,
            clean_img_conf=self.clean_img_dict,
            conf_threshold=self.config.conf_threshold)

        self.noise_amount = NoiseAmount(self.config.radius_lower_bound, self.config.radius_upper_bound)

        # self.set_multiple_gpu()
        self.set_to_device()

        if self.config.eval_data_type == 'ordered' or self.config.eval_data_type == 'one':
            split_dataset = SplitDataset(
                img_dir=self.config.img_dir,
                lab_dir=self.config.lab_dir,
                max_lab=self.config.max_labels_per_img,
                img_size=self.img_size,
                transform=transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]))
        else:
            split_dataset = SplitDataset1(
                img_dir_train_val=self.config.img_dir,
                lab_dir_train_val=self.config.lab_dir,
                img_dir_test=self.config.img_dir_test,
                lab_dir_test=self.config.lab_dir_test,
                max_lab=self.config.max_labels_per_img,
                img_size=self.img_size,
                transform=transforms.Compose(
                    [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()]))

        self.train_loader, self.val_loader, self.test_loader = split_dataset(val_split=0.2,
                                                                             test_split=0.2,
                                                                             shuffle_dataset=True,
                                                                             random_seed=seed,
                                                                             batch_size=self.config.batch_size,
                                                                             ordered=True)

        self.train_losses = []
        self.val_losses = []

        self.max_prob_losses = []
        self.cor_det_losses = []
        self.nps_losses = []
        self.noise_losses = []

        self.train_acc = []
        self.val_acc = []
        self.final_epoch_count = self.config.epochs

        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        if 'SLURM_JOBID' not in os.environ.keys():
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S")
        else:
            self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + os.environ['SLURM_JOBID']
        self.create_folders()
        # self.save_config_details()

        # subprocess.Popen(['tensorboard', '--logdir=' + self.current_dir + '/runs'])
        # self.writer = SummaryWriter(self.current_dir + '/runs')
        self.writer = None

    def set_to_device(self):
        self.dot_applier = self.dot_applier.to(device)
        self.patch_applier = self.patch_applier.to(device)
        self.detections = self.detections.to(device)
        self.non_printability_score = self.non_printability_score.to(device)
        self.noise_amount = self.noise_amount.to(device)

    def set_multiple_gpu(self):
        if torch.cuda.device_count() > 1:
            print("more than 1")
            self.dot_applier = DP(self.dot_applier)
            self.patch_applier = DP(self.patch_applier)
            self.detections = DP(self.detections)

    def create_folders(self):
        Path('/'.join(self.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/testing').mkdir(parents=True, exist_ok=True)

    def train(self):
        epoch_length = len(self.train_loader)
        print(f'One epoch is {epoch_length} batches', flush=True)

        optimizer = Adam([{'params': self.dot_applier.theta, 'lr': self.config.loc_lr},
                          {'params': self.dot_applier.colors, 'lr': self.config.color_lr},
                          {'params': self.dot_applier.radius, 'lr': self.config.radius_lr}],
                         amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(delta=1e-3, current_dir=self.current_dir, patience=20)

        clipper = WeightClipper(self.config.radius_lower_bound, self.config.radius_upper_bound)
        adv_patch_cpu = torch.zeros((1, 3, self.img_size, self.img_size), dtype=torch.float32)
        alpha_cpu = torch.zeros((1, 1, self.img_size, self.img_size), dtype=torch.float32)
        for epoch in range(self.config.epochs):
            train_loss = 0.0
            max_prob_loss = 0.0
            cor_det_loss = 0.0
            nps_loss = 0.0
            noise_loss = 0.0

            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, ' \
                            'maxprob-loss: {:.6}, ' \
                            'corr det-loss: {:.6}, ' \
                            'nps-loss: {:.6}, ' \
                            'noise-loss: {:.6}'
            for i_batch, (img_batch, lab_batch, img_names) in progress_bar:
                # move tensors to gpu
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                adv_patch = adv_patch_cpu.to(device)
                alpha = alpha_cpu.to(device)

                # forward prop
                adv_patch, alpha = self.dot_applier(adv_patch, alpha)  # put dots on patch

                applied_batch = self.patch_applier(img_batch, adv_patch, alpha)  # apply patch on a batch of images

                if epoch == 0 and i_batch == 0:
                    self.save_initial_patch(adv_patch, alpha)

                output_patch = self.yolov5(applied_batch)[0]  # get yolo output with patch

                max_prob, cor_det = self.detections(lab_batch, output_patch, img_names)

                nps = self.non_printability_score(self.dot_applier.colors)

                noise = self.noise_amount(self.dot_applier.radius)

                loss, loss_arr = self.loss_function(max_prob, cor_det, nps, noise)  # calculate loss

                # save losses
                max_prob_loss += loss_arr[0].item()
                cor_det_loss += loss_arr[1].item()
                nps_loss += loss_arr[2].item()
                noise_loss += loss_arr[3].item()
                train_loss += loss.item()

                # back prop
                optimizer.zero_grad()
                loss.backward()

                # update parameters
                optimizer.step()
                self.dot_applier.apply(clipper)  # clip x,y coordinates

                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  max_prob_loss / (i_batch + 1),
                                                                  cor_det_loss / (i_batch + 1),
                                                                  nps_loss / (i_batch + 1),
                                                                  noise_loss / (i_batch + 1)))

                if i_batch % 1 == 0 and self.writer is not None:
                    self.write_to_tensorboard(adv_patch, alpha,train_loss, max_prob_loss, cor_det_loss, nps_loss, noise_loss,
                                              epoch_length, epoch, i_batch, optimizer)
                # self.writer.add_image('patch', adv_patch.squeeze(0), epoch_length * epoch + i_batch)
                if i_batch + 1 == epoch_length:
                    self.last_batch_calc(adv_patch, alpha, epoch_length, progress_bar, prog_bar_desc,
                                         train_loss, max_prob_loss, cor_det_loss, nps_loss, noise_loss,
                                         optimizer, epoch, i_batch)

                # self.run_slide_show(adv_patch)

                # clear gpu
                del img_batch, lab_batch, applied_batch, output_patch, max_prob, cor_det, nps, noise, loss
                torch.cuda.empty_cache()

            # check if loss has decreased
            if early_stop(self.val_losses[-1], adv_patch.cpu(), alpha.cpu(), epoch):
                self.final_epoch_count = epoch
                break

            scheduler.step(self.val_losses[-1])

        self.adv_patch = early_stop.best_patch
        self.alpha = early_stop.best_alpha
        print("Training finished")

    def get_image_size(self):
        if type(self.yolov2) == nn.DataParallel:
            img_size = self.yolov2.module.height
        else:
            img_size = self.yolov2.height
        return int(img_size)

    def evaluate_loss(self, loader, adv_patch, alpha):
        total_loss = 0.0
        for img_batch, lab_batch, img_names in loader:
            with torch.no_grad():
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)

                applied_batch = self.patch_applier(img_batch, adv_patch, alpha)
                output_patch = self.yolov5(applied_batch)[0]
                max_prob, cor_det = self.detections(lab_batch, output_patch, img_names)
                nps = self.non_printability_score(self.dot_applier.colors)
                noise = self.noise_amount(self.dot_applier.radius)
                batch_loss, _ = self.loss_function(max_prob, cor_det, nps, noise)
                total_loss += batch_loss.item()

                del img_batch, lab_batch, applied_batch, output_patch, max_prob, cor_det, nps, noise, batch_loss
                torch.cuda.empty_cache()
        loss = total_loss / len(loader)
        return loss

    def plot_train_val_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        plt.plot(epochs, self.train_losses, 'b', label='Training loss')
        plt.plot(epochs, self.val_losses, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir + '/final_results/train_val_loss_plt.png')
        plt.close()

    def plot_separate_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        weights = np.array([self.config.max_prob_weight, self.config.pres_det_weight, self.config.nps_weight, self.config.noise_weight])
        number_of_subplots = weights[weights > 0].astype(np.bool).sum()
        fig, axes = plt.subplots(nrows=1, ncols=number_of_subplots, figsize=(5 * number_of_subplots, 3 * number_of_subplots), squeeze=False)
        for idx, (weight, loss, label, color_name) in enumerate(zip([self.config.max_prob_weight, self.config.pres_det_weight, self.config.nps_weight, self.config.noise_weight],
                                                                    [self.max_prob_losses, self.cor_det_losses, self.nps_losses, self.noise_losses],
                                                                    ['Max probability loss', 'Correct detections loss', 'Non printability loss', 'Noise Amount loss'],
                                                                    'brgkyc')):
            if weight > 0:
                axes[0, idx].plot(epochs, loss, c=color_name, label=label)
                axes[0, idx].set_xlabel('Epoch')
                axes[0, idx].set_ylabel('Loss')
                axes[0, idx].legend(loc='upper right')
        fig.tight_layout()
        plt.savefig(self.current_dir + '/final_results/separate_loss_plt.png')
        plt.close()

    def plot_combined(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        fig, ax1 = plt.subplots(ncols=1, figsize=(8, 4))
        ax1.plot(epochs, self.max_prob_losses, c='b', label='Max Probability')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(epochs, self.cor_det_losses, c='r', label='Correct Detections')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        plt.savefig(self.current_dir + '/final_results/combined_losses.png')
        plt.close()

    def loss_function(self, max_prob, cor_det, nps, noise):
        max_prob_loss = self.config.max_prob_weight * torch.mean(max_prob)
        cor_det_loss = self.config.pres_det_weight * torch.mean(cor_det)
        nps_loss = self.config.nps_weight * nps
        noise_loss = self.config.noise_weight * noise
        return max_prob_loss + cor_det_loss + nps_loss + noise_loss, [max_prob_loss, cor_det_loss, nps_loss, noise_loss]

    def save_final_objects(self):
        # save patch
        transforms.ToPILImage()(self.adv_patch.squeeze(0)).save(
            self.current_dir + '/final_results/final_patch_wo_alpha.png', 'PNG')
        torch.save(self.adv_patch, self.current_dir + '/final_results/final_patch_raw.pt')
        transforms.ToPILImage()(self.alpha.squeeze(0)).save(self.current_dir + '/final_results/alpha.png', 'PNG')
        torch.save(self.alpha, self.current_dir + '/final_results/alpha_raw.pt')
        final_patch = torch.cat([self.adv_patch.squeeze(0), self.alpha.squeeze(0)])
        transforms.ToPILImage()(final_patch.cpu()).save(self.current_dir + '/final_results/final_patch_w_alpha.png',
                                                        'PNG')
        # save losses
        with open(self.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses, fp)
        with open(self.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.current_dir + '/losses/max_prob_losses', 'wb') as fp:
            pickle.dump(self.max_prob_losses, fp)
        with open(self.current_dir + '/losses/cor_det_losses', 'wb') as fp:
            pickle.dump(self.cor_det_losses, fp)
        with open(self.current_dir + '/losses/nps_losses', 'wb') as fp:
            pickle.dump(self.nps_losses, fp)
        with open(self.current_dir + '/losses/noise_losses', 'wb') as fp:
            pickle.dump(self.noise_losses, fp)

    def save_final_results(self, avg_precision):
        target_noise_ap, target_patch_ap, other_noise_ap, other_patch_ap = avg_precision
        # calculate test loss
        test_loss = self.evaluate_loss(self.test_loader, self.adv_patch.to(device),
                                       self.alpha.to(device))
        print("Test loss: " + str(test_loss))
        self.save_config_details()
        row_to_csv = \
            str(self.train_losses[-1]) + ',' + \
            str(self.val_losses[-1]) + ',' + \
            str(test_loss) + ',' + \
            str(self.max_prob_losses[-1]) + ',' + \
            str(self.cor_det_losses[-1]) + ',' + \
            str(self.nps_losses[-1]) + ',' + \
            str(self.final_epoch_count) + '/' + str(self.config.epochs) + ',' + \
            str(target_noise_ap) + ',' + \
            str(target_patch_ap) + ',' + \
            str(other_noise_ap) + ',' + \
            str(other_patch_ap) + '\n'

        # write results to csv
        with open('experiments/results.csv', 'a') as fd:
            fd.write(row_to_csv)

    def write_to_tensorboard(self, adv_patch, alpha, train_loss, max_prob_loss, cor_det_loss, nps_loss, noise_loss,
                             epoch_length, epoch, i_batch, optimizer):
        iteration = epoch_length * epoch + i_batch
        self.writer.add_scalar('train_loss', train_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/max_prob_loss', max_prob_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/cor_det_loss', cor_det_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/nps_loss', nps_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/noise_loss', noise_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('misc/epoch', epoch, iteration)
        self.writer.add_scalar('misc/loc_learning_rate', optimizer.param_groups[0]["lr"], iteration)
        self.writer.add_scalar('misc/color_learning_rate', optimizer.param_groups[1]["lr"], iteration)
        self.writer.add_scalar('misc/radius_learning_rate', optimizer.param_groups[2]["lr"], iteration)
        self.writer.add_image('patch_rgb', adv_patch.squeeze(0), iteration)
        self.writer.add_image('patch_rgba', torch.cat([adv_patch.squeeze(0), alpha.squeeze(0)]), iteration)

    def last_batch_calc(self, adv_patch, alpha, epoch_length, progress_bar, prog_bar_desc,
                        train_loss, max_prob_loss, cor_det_loss, nps_loss, noise_loss,
                        optimizer, epoch, i_batch):
        # calculate epoch losses
        train_loss /= epoch_length
        max_prob_loss /= epoch_length
        cor_det_loss /= epoch_length
        nps_loss /= epoch_length
        noise_loss /= epoch_length
        self.train_losses.append(train_loss)
        self.max_prob_losses.append(max_prob_loss)
        self.cor_det_losses.append(cor_det_loss)
        self.nps_losses.append(nps_loss)
        self.noise_losses.append(noise_loss)

        # check on validation
        val_loss = self.evaluate_loss(self.val_loader, adv_patch, alpha)
        self.val_losses.append(val_loss)

        prog_bar_desc += ', val-loss: {:.6}, loc-lr: {:.6}, color-lr: {:.6}, radius-lr: {:.6}'
        progress_bar.set_postfix_str(prog_bar_desc.format(train_loss,
                                                          max_prob_loss,
                                                          cor_det_loss,
                                                          nps_loss,
                                                          noise_loss,
                                                          val_loss,
                                                          optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[1]['lr'],
                                                          optimizer.param_groups[2]['lr']))
        if self.writer is not None:
            self.writer.add_scalar('loss/val_loss', val_loss, epoch_length * epoch + i_batch)

    def get_clean_image_conf(self):
        clean_img_dict = dict()
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for img_batch, lab_batch, img_name in loader:
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)

                output = self.yolov5(img_batch)[0]
                output = output.transpose(1, 2).contiguous()
                output_objectness, output = output[:, 4, :], output[:, 5:, :]
                batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
                for i in range(batch_idx.size()[0]):
                    ids = np.unique(
                        batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(
                            int))
                    if len(ids) == 0:
                        continue
                    clean_img_dict[img_name[i]] = dict()
                    # get relevant classes
                    confs_for_class = output[i, ids, :]
                    confs_if_object = self.config.loss_target(output_objectness[i], confs_for_class)

                    # find the max prob for each related class
                    max_conf, _ = torch.max(confs_if_object, dim=1)
                    for j, label in enumerate(ids):
                        clean_img_dict[img_name[i]][label] = max_conf[j].item()

                del img_batch, lab_batch, output, output_objectness, batch_idx
                torch.cuda.empty_cache()

        print(len(clean_img_dict))
        np.save('confidences/' + self.config.model_name + '/medium/clean_img_conf_lisa_ordered.npy', clean_img_dict)

    # def get_clean_image_conf(self):
    #     clean_img_dict = dict()
    #     for loader in [self.train_loader, self.val_loader, self.test_loader]:
    #         for img_batch, lab_batch, img_name in loader:
    #             img_batch = img_batch.to(device)
    #             lab_batch = lab_batch.to(device)
    #
    #             output = self.yolov2(img_batch)
    #             batch = output.size(0)
    #             h = output.size(2)
    #             w = output.size(3)
    #             output = output.view(batch, self.yolov2.num_anchors, 5 + self.config.num_classes,
    #                                  h * w)  # [batch, 5, 85, 361]
    #             output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
    #             output = output.view(batch, 5 + self.config.num_classes,
    #                                  self.yolov2.num_anchors * h * w)  # [batch, 85, 1805]
    #             output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
    #             output = output[:, 5:5 + self.config.num_classes, :]  # [batch, 80, 1805]
    #             normal_confs = torch.nn.Softmax(dim=1)(output)  # [batch, 80, 1805]
    #             batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
    #             for i in range(batch_idx.size(0)):
    #                 ids = np.unique(
    #                     batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(
    #                         int))
    #                 if len(ids) == 0:
    #                     continue
    #                 clean_img_dict[img_name[i]] = dict()
    #                 # get relevant classes
    #                 confs_for_class = normal_confs[i, ids, :]
    #                 confs_if_object = self.config.loss_target(output_objectness[i], confs_for_class)
    #
    #                 # find the max prob for each related class
    #                 max_conf, _ = torch.max(confs_if_object, dim=1)
    #                 for j, label in enumerate(ids):
    #                     clean_img_dict[img_name[i]][label] = max_conf[j].item()
    #
    #             del img_batch, lab_batch, output, output_objectness, normal_confs, batch_idx
    #             torch.cuda.empty_cache()
    #
    #     print(len(clean_img_dict))
    #     np.save('confidences/clean_img_conf_lisa_new_color.npy', clean_img_dict)

    def save_config_details(self):
        # write results to csv
        row_to_csv = self.current_dir.split('/')[-1] + ',' + \
                     self.config.model_name + ',' + \
                     self.config.img_dir.split('/')[-2] + ',' + \
                     str(self.config.loc_lr) + '-' + str(self.config.color_lr) + '-' + str(self.config.radius_lr) + ',' + \
                     str(self.config.sched_cooldown) + ',' + \
                     str(self.config.sched_patience) + ',' + \
                     str(self.config.loss_mode) + ',' + \
                     str(self.config.conf_threshold) + ',' + \
                     str(self.config.max_prob_weight) + ',' + \
                     str(self.config.pres_det_weight) + ',' + \
                     str(self.config.nps_weight) + ',' + \
                     str(self.config.num_of_dots) + ',' + \
                     str(None) + ',' + \
                     str(self.config.alpha_max) + ',' + \
                     str(self.config.beta_dropoff) + ','
        with open('experiments/results.csv', 'a') as fd:
            fd.write(row_to_csv)

    def save_initial_patch(self, adv_patch, alpha):
        transforms.ToPILImage()(adv_patch.cpu().squeeze(0)).save(self.current_dir + '/saved_patches/initial_patch.png')
        transforms.ToPILImage()(alpha.cpu().squeeze(0)).save(self.current_dir + '/saved_patches/initial_alpha.png')

    @staticmethod
    def run_slide_show(adv_patch):
        adv_to_show = adv_patch.detach().cpu()
        adv_to_show = torch.where(adv_to_show == 0, torch.ones_like(adv_to_show), adv_to_show)
        transforms.ToPILImage()(adv_to_show.squeeze(0)).save('current_slide.jpg')
        img = cv2.imread('current_slide.jpg')
        cv2.imshow('slide show', img)
        cv2.waitKey(1)


def main():
    mode = 'private'
    # mode = 'cluster'
    patch_train = TrainPatch(mode)
    # patch_train.get_clean_image_conf()
    patch_train.train()
    patch_train.save_final_objects()
    patch_train.plot_train_val_loss()
    patch_train.plot_separate_loss()
    patch_train.plot_combined()
    patch_eval = EvaluateYOLO(patch_train.current_dir, patch_train.test_loader, patch_train.patch_applier, patch_train.yolov5, patch_train.config.class_id, patch_train.config.conf_threshold, patch_train.config.iou_threshold, patch_train.config.eval_data_type)
    # patch_eval.create_yolo_true_labels()
    avg_precision = patch_eval.calculate()
    patch_train.save_final_results(avg_precision)
    print('Writing final results finished', flush=True)


if __name__ == '__main__':
    main()
