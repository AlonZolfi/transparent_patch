import sys
import os

if sys.base_prefix.__contains__('home/zolfi'):
    sys.path.append('/home/zolfi/transparent_patch/patch')
    sys.path.append('/home/zolfi/transparent_patch/pytorch-yolo2')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
else:
    from torchviz import make_dot

import torch
from torch import nn
from torch.nn.parallel import DataParallel as DP
from torch.optim import Adam
from tensorboardX import SummaryWriter

from torchvision import transforms

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path
import subprocess
import pickle
import numpy as np

from config import patch_config_types
from darknet import Darknet
from load_data import SplitDataset
from nn_modules import PatchApplier, DotApplier, TotalVariation, PreserveDetections
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

        self.dot_applier = DotApplier(
            self.config.num_of_dots,
            self.get_image_size())

        self.patch_applier = PatchApplier()

        # self.prob_extractor = MaxProbExtractor(
        #     weight=self.config.max_prob_weight,
        #     cls_id=self.config.class_id,
        #     num_cls=self.config.num_classes,
        #     config=self.config,
        #     num_anchor=self.yolo.num_anchors)

        self.clean_img_dict = np.load('clean_img_conf.npy', allow_pickle=True).item()
        self.preserve_detec = PreserveDetections(
            weight_cls=self.config.max_prob_weight,
            weight_others=self.config.pres_det_weight,
            cls_id=self.config.class_id,
            num_cls=self.config.num_classes,
            config=self.config,
            num_anchor=self.yolo.num_anchors,
            clean_img_conf=self.clean_img_dict
        )
        self.total_variation = TotalVariation(weight=self.config.tv_weight)
        # self.iou = IoU(weight=self.config.iou_weight)
        # self.patch_transformer = PatchTransformer()
        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size)

        self.max_tv = torch.tensor(self.config.max_tv).to(device)

        img_size = self.yolo.height
        self.alpha_tensor_cpu = torch.full((1, 1, img_size, img_size),
                                           dtype=torch.float32,
                                           fill_value=0.4,
                                           requires_grad=True)
        self.adv_patch_cpu = torch.full((1, 3, img_size, img_size), dtype=torch.float32, fill_value=1)

        self.set_multiple_gpu()
        self.set_to_device()

        split_dataset = SplitDataset(img_dir=self.config.img_dir,
                                     lab_dir=self.config.lab_dir,
                                     max_lab=self.config.max_labels_per_img,
                                     img_size=img_size,
                                     transform=transforms.Compose([transforms.Resize((img_size, img_size)),
                                                                   transforms.ToTensor()]))
        self.train_loader, self.val_loader, self.test_loader = split_dataset(val_split=0.2,
                                                                             test_split=0.2,
                                                                             shuffle_dataset=True,
                                                                             random_seed=SEED,
                                                                             batch_size=self.config.batch_size)

        self.train_losses = []
        self.val_losses = []
        self.max_prob_losses = []
        self.cor_det_losses = []
        self.tv_losses = []
        self.train_acc = []
        self.val_acc = []
        self.final_epochs = self.config.epochs

        if 'SLURM_JOBID' not in os.environ.keys():
            self.current_dir = "experiments/" + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S")
        else:
            self.current_dir = "experiments/" + time.strftime("%d-%m-%Y") + '_' + os.environ['SLURM_JOBID']
        self.create_folders()

        # subprocess.Popen(['tensorboard', '--logdir=' + self.current_dir + '/runs'])
        # self.writer = SummaryWriter(self.current_dir + '/runs')
        self.writer = None

        torch.manual_seed(SEED)

    def set_to_device(self):
        self.yolo = self.yolo.to(device)
        self.dot_applier = self.dot_applier.to(device)
        self.patch_applier = self.patch_applier.to(device)
        self.total_variation = self.total_variation.to(device)
        self.preserve_detec = self.preserve_detec.to(device)
        # self.iou.to(device)
        # self.patch_transformer.to(device)
        # self.nps_calculator.to(device)

    def set_multiple_gpu(self):
        if torch.cuda.device_count() > 1:
            print("more than 1")
            self.yolo = DP(self.yolo)
            self.dot_applier = DP(self.dot_applier)
            self.patch_applier = DP(self.patch_applier)
            self.total_variation = DP(self.total_variation)
            self.preserve_detec = DP(self.preserve_detec)
            # self.adv_patch_cpu = self.adv_patch_cpu.unsqueeze(0)
            # self.iou = torch.nn.DataParallel(self.iou)
            # self.patch_transformer = torch.nn.DataParallel(self.patch_transformer)
            # self.nps_calculator = torch.nn.DataParallel(self.nps_calculator)

    def create_folders(self):
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/acc').mkdir(parents=True, exist_ok=True)

    def train(self):
        epoch_length = len(self.train_loader)
        print(f'One epoch is {epoch_length} batches', flush=True)

        optimizer = Adam(self.dot_applier.parameters(),
                         # optimizer = Adam([self.adv_patch_cpu, self.alpha_tensor_cpu],
                         lr=self.config.start_learning_rate,
                         amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        early_stop = EarlyStopping(delta=1e-5, current_dir=self.current_dir, patience=50)

        for epoch in range(self.config.epochs):
            max_prob_loss = 0.0
            tv_loss = 0.0
            cor_det_loss = 0.0
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0

            progress_bar = tqdm(enumerate(self.train_loader), desc=f'Epoch {epoch}', total=epoch_length)
            prog_bar_desc = 'train-loss: {:.6}, train-acc: {:.6}, maxprob-loss: {:.6}, tv-loss: {:.6}, corr det-loss: {:.6}'
            for i_batch, (img_batch, lab_batch, img_names) in progress_bar:
                # move tensors to cuda
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                adv_patch = self.adv_patch_cpu.to(device)
                alpha_tensor = self.alpha_tensor_cpu.to(device)

                # forward prop
                adv_patch = self.dot_applier(adv_patch)  # update patch

                # apply patch on a batch of images
                applied_batch = self.patch_applier(img_batch, adv_patch, alpha_tensor)

                output_patch = self.yolo(applied_batch)  # get yolo output with patch

                # max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, output_clean)
                max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, img_names)
                tv = self.total_variation(adv_patch)  # calculate patch total variation

                # calculate loss
                loss, loss_arr = self.loss_function(max_prob, tv, cor_det)

                # save losses
                max_prob_loss += loss_arr[0].item()
                tv_loss += loss_arr[1].item()
                cor_det_loss += loss_arr[2].item()
                train_loss += loss.item()

                train_acc += self.accuracy_score(lab_batch, output_patch, img_names)

                # back prop
                optimizer.zero_grad()
                loss.backward()

                # update parameters
                optimizer.step()

                # keep patch values within (0,1) range
                # self.adv_patch_cpu = torch.clamp(self.adv_patch_cpu, 0.000001, 0.999999)

                # make_dot(loss).render("prop_path/graph"+str(i_batch), format="png")
                progress_bar.set_postfix_str(prog_bar_desc.format(train_loss / (i_batch + 1),
                                                                  train_acc / (i_batch + 1),
                                                                  max_prob_loss / (i_batch + 1),
                                                                  tv_loss / (i_batch + 1),
                                                                  cor_det_loss / (i_batch + 1)))
                if i_batch % 10 == 0 and self.writer is not None:
                    self.write_to_tensorboard(train_loss, train_acc, max_prob_loss, tv_loss, cor_det_loss,
                                              epoch_length, epoch, i_batch, optimizer)

                if i_batch + 1 == epoch_length:
                    self.last_batch_calc(epoch_length, progress_bar, prog_bar_desc,
                                         train_loss, max_prob_loss, tv_loss, cor_det_loss, train_acc,
                                         optimizer, epoch, i_batch)

                # clear gpu
                del img_batch, lab_batch, adv_patch, alpha_tensor, applied_batch, \
                    output_patch, max_prob, cor_det, tv, loss
                torch.cuda.empty_cache()

            # check if loss has decreased
            if early_stop(self.val_losses[-1], self.adv_patch_cpu, epoch):
                self.final_epochs = epoch
                break

            scheduler.step(val_loss)

        print("Training finished")

    def get_image_size(self):
        if type(self.yolo) == nn.DataParallel:
            img_size = self.yolo.module.height
        else:
            img_size = self.yolo.height
        return int(img_size)

    def evaluate_loss_and_acc(self, loader):
        total_loss = 0.0
        total_acc = 0.0
        for img_batch, lab_batch, img_names in loader:
            with torch.no_grad():
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)
                adv_patch = self.adv_patch_cpu.to(device)
                alpha = self.alpha_tensor_cpu.to(device)
                applied_batch = self.patch_applier(img_batch, adv_patch, alpha)
                output_patch = self.yolo(applied_batch)
                # output_clean = self.yolo(img_batch)
                tv = self.total_variation(adv_patch)
                # max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, output_clean)
                max_prob, cor_det = self.preserve_detec(lab_batch, output_patch, img_names)
                batch_loss, _ = self.loss_function(max_prob, tv, cor_det)
                total_loss += batch_loss.item()
                total_acc += self.accuracy_score(lab_batch, output_patch, img_names)
                del img_batch, lab_batch, adv_patch, alpha, applied_batch, \
                    output_patch, tv, max_prob, cor_det
                torch.cuda.empty_cache()
        loss = total_loss / len(loader)
        acc = total_acc / len(loader)
        return loss, acc

    def accuracy_score(self, lab_batch, output, img_names):
        def get_correct_ids():
            ids = np.unique(
                batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(int))
            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_dict[img_names[i]][lab_id])
            max_clean = torch.tensor(clean_img_values, device=device)
            clean_gt = (max_clean > conf_threshold).cpu().numpy()
            return np.array(ids)[clean_gt].tolist()
        not_attacked_acc = []
        conf_threshold = 0.5
        batch = output.size(0)
        h = output.size(2)
        w = output.size(3)
        output = output.view(batch, self.yolo.num_anchors, 5 + self.config.num_classes, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.config.num_classes, self.yolo.num_anchors * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.config.num_classes, :]  # [batch, 80, 1805]
        normal_confs = torch.nn.Softmax(dim=1)(output)  # [batch, 80, 1805]
        confs_for_attacked_class = normal_confs[:, self.config.class_id, :]  # [batch, 1805]

        batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
        for i in range(batch_idx.size(0)):
            ids = get_correct_ids()
            if len(ids) == 0:
                not_attacked_acc.append(1)
                continue
            # get relevant classes
            confs_for_class = normal_confs[i, ids, :]
            confs_if_object_other = self.config.loss_target(output_objectness[i], confs_for_class)

            # find the max prob for each related class
            max_conf_other, _ = torch.max(confs_if_object_other, dim=1)
            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_dict[img_names[i]][lab_id])

            not_attacked_acc.append(int((max_conf_other > conf_threshold).sum().item() / torch.numel(max_conf_other)))

        confs_if_object = self.config.loss_target(output_objectness, confs_for_attacked_class)  # [batch, 1805]
        # find the max probability for stop sign
        max_conf, _ = torch.max(confs_if_object, dim=1)  # [batch]
        attacked_acc = ((~(max_conf > conf_threshold)).int()).detach().cpu().numpy()
        acc = np.mean((0.5 * attacked_acc) + (0.5 * np.array(not_attacked_acc)))
        return acc

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

    def plot_train_val_acc(self):
        epochs = [x + 1 for x in range(len(self.train_acc))]
        plt.plot(epochs, self.train_acc, 'b', label='Training accuracy')
        plt.plot(epochs, self.val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir + '/final_results/train_val_acc_plt.png')
        plt.close()

    def plot_separate_loss(self):
        epochs = [x + 1 for x in range(len(self.train_losses))]
        plt.plot(epochs, self.max_prob_losses, 'b', label='Max probability loss')
        plt.plot(epochs, self.cor_det_losses, 'g', label='Correct detections loss')
        plt.plot(epochs, self.tv_losses, 'r', label='Total variation loss')
        plt.title('Separate losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(self.current_dir + '/final_results/separate_loss_plt.png')
        plt.close()

    def loss_function(self, max_prob, tv, cor_det):
        max_prob_loss = torch.mean(max_prob)
        tv_loss = torch.max(tv, self.max_tv)
        cor_det_loss = torch.mean(cor_det)
        return max_prob_loss + tv_loss + cor_det_loss, [max_prob_loss, tv_loss, cor_det_loss]

    def save_final_results(self):
        # save patch
        transforms.ToPILImage()(self.adv_patch_cpu.squeeze(0)).save(self.current_dir + '/final_results/final_patch_wo_alpha.png', 'PNG')
        transforms.ToPILImage()(self.alpha_tensor_cpu.squeeze(0)).save(self.current_dir + '/final_results/alpha_channel.png', 'PNG')
        final_patch = torch.cat([self.adv_patch_cpu.squeeze(0), self.alpha_tensor_cpu.squeeze(0)])
        transforms.ToPILImage()(final_patch.cpu()).save(self.current_dir + '/final_results/final_patch_w_alpha.png', 'PNG')
        # save losses
        with open(self.current_dir + '/losses/train_losses', 'wb') as fp:
            pickle.dump(self.train_losses, fp)
        with open(self.current_dir + '/losses/val_losses', 'wb') as fp:
            pickle.dump(self.val_losses, fp)
        with open(self.current_dir + '/losses/max_prob_losses', 'wb') as fp:
            pickle.dump(self.max_prob_losses, fp)
        with open(self.current_dir + '/losses/cor_det_losses', 'wb') as fp:
            pickle.dump(self.cor_det_losses, fp)
        with open(self.current_dir + '/losses/tv_losses', 'wb') as fp:
            pickle.dump(self.tv_losses, fp)
        with open(self.current_dir + '/acc/train_acc', 'wb') as fp:
            pickle.dump(self.train_acc, fp)
        with open(self.current_dir + '/acc/val_acc', 'wb') as fp:
            pickle.dump(self.val_acc, fp)
        # calculate test loss
        test_loss, test_acc = self.evaluate_loss_and_acc(self.test_loader)
        print("Test loss: " + str(test_loss))
        print("Test acc: " + str(test_acc))
        row_to_csv = self.current_dir.split('/')[-1] + ',' + \
                     str(self.train_losses[-1]) + ',' + \
                     str(self.val_losses[-1]) + ',' + \
                     str(test_loss) + ',' + \
                     str(self.train_acc[-1]) + ',' + \
                     str(self.val_acc[-1]) + ',' + \
                     str(test_acc) + ',' + \
                     str(self.max_prob_losses[-1]) + ',' + \
                     str(self.cor_det_losses[-1]) + ',' + \
                     str(self.tv_losses[-1]) + ',' + \
                     str(self.final_epochs) + '\n'
        # write results to csv
        with open('experiments/results.csv', 'a') as fd:
            fd.write(row_to_csv)

    def write_to_tensorboard(self, train_loss, train_acc, max_prob_loss, tv_loss,
                             cor_det_loss, epoch_length, epoch, i_batch, optimizer):
        iteration = epoch_length * epoch + i_batch
        self.writer.add_scalar('train_loss', train_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('train_acc', train_acc / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/max_prob_loss', max_prob_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/tv_loss', tv_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('loss/cor_det_loss', cor_det_loss / (i_batch + 1), iteration)
        self.writer.add_scalar('misc/epoch', epoch, iteration)
        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
        self.writer.add_image('patch', self.adv_patch_cpu, iteration)

    def last_batch_calc(self, epoch_length, progress_bar, prog_bar_desc,
                        train_loss, max_prob_loss, tv_loss, cor_det_loss, train_acc,
                        optimizer, epoch, i_batch):
        # calculate epoch losses
        train_loss /= epoch_length
        max_prob_loss /= epoch_length
        tv_loss /= epoch_length
        cor_det_loss /= epoch_length
        self.train_losses.append(train_loss)
        self.max_prob_losses.append(max_prob_loss)
        self.tv_losses.append(tv_loss)
        self.cor_det_losses.append(cor_det_loss)

        train_acc /= epoch_length
        self.train_acc.append(train_acc)

        # check on validation
        val_loss, val_acc = self.evaluate_loss_and_acc(self.val_loader)
        self.val_losses.append(val_loss)
        self.val_acc.append(val_acc)

        prog_bar_desc += ', val-loss: {:.6}, val-acc: {:.6}, lr: {:.6}'
        progress_bar.set_postfix_str(prog_bar_desc.format(train_loss,
                                                          train_acc,
                                                          max_prob_loss,
                                                          tv_loss,
                                                          cor_det_loss,
                                                          val_loss,
                                                          val_acc,
                                                          optimizer.param_groups[0]['lr']))
        if self.writer is not None:
            self.writer.add_scalar('val_loss', val_loss, epoch_length * epoch + i_batch)
            self.writer.add_scalar('val_acc', val_acc, epoch_length * epoch + i_batch)

    def get_clean_image_conf(self):
        clean_img_dict = dict()
        for loader in [self.train_loader, self.val_loader, self.test_loader]:
            for img_batch, lab_batch, img_name in loader:
                img_batch = img_batch.to(device)
                lab_batch = lab_batch.to(device)

                output = self.yolo(img_batch)
                batch = output.size(0)
                h = output.size(2)
                w = output.size(3)
                output = output.view(batch, self.yolo.num_anchors, 5 + self.config.num_classes, h * w)  # [batch, 5, 85, 361]
                output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
                output = output.view(batch, 5 + self.config.num_classes, self.yolo.num_anchors * h * w)  # [batch, 85, 1805]
                output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
                output = output[:, 5:5 + self.config.num_classes, :]  # [batch, 80, 1805]
                normal_confs = torch.nn.Softmax(dim=1)(output)  # [batch, 80, 1805]
                batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
                for i in range(batch_idx.size(0)):
                    ids = np.unique(
                        batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(int))
                    if len(ids) == 0:
                        continue
                    clean_img_dict[img_name[i]] = dict()
                    # get relevant classes
                    confs_for_class = normal_confs[i, ids, :]
                    confs_if_object = self.config.loss_target(output_objectness[i], confs_for_class)

                    # find the max prob for each related class
                    max_conf, _ = torch.max(confs_if_object, dim=1)
                    for j, label in enumerate(ids):
                        clean_img_dict[img_name[i]][label] = max_conf[j].item()

                del img_batch, lab_batch, output, output_objectness, normal_confs, batch_idx
                torch.cuda.empty_cache()

        # with open('clean_img_conf.txt', 'w') as file:
        #     file.write(json.dumps(clean_img_dict))
        print(len(clean_img_dict))
        np.save('clean_img_conf.npy', clean_img_dict)


def main():
    mode = 'private'
    # mode = 'cluster'
    patch_train = TrainPatch(mode)
    patch_train.train()
    patch_train.save_final_results()
    patch_train.plot_train_val_loss()
    patch_train.plot_train_val_acc()
    patch_train.plot_separate_loss()
    print('Writing final results finished', flush=True)


if __name__ == '__main__':
    main()
