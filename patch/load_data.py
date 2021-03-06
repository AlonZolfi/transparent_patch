import fnmatch
import os

import numpy as np

from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from torchvision import transforms
import matplotlib.pyplot as plt
import random


class LisaDataset(Dataset):
    def __init__(self, img_dir, lab_dir, max_lab, img_size, shuffle=True, transform=None):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_size = img_size
        self.shuffle = shuffle
        self.img_names = self.get_image_names()
        self.img_paths = self.get_image_paths()
        self.lab_paths = self.get_lab_paths()
        self.max_n_labels = max_lab
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        if os.path.getsize(lab_path):  # check to see if label file contains data.
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        # image, label = self.pad_and_scale(image, label)
        if self.transform:
            image = self.transform(image)

        label = self.pad_lab(label)
        return image, label, self.img_names[idx]

    def get_image_names(self):
        png_images = fnmatch.filter(os.listdir(self.img_dir), '*.png')
        jpg_images = fnmatch.filter(os.listdir(self.img_dir), '*.jpg')
        n_png_images = len(png_images)
        n_jpg_images = len(jpg_images)
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(self.lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        return png_images + jpg_images

    def get_image_paths(self):
        img_paths = []
        for img_name in self.img_names:
            img_paths.append(os.path.join(self.img_dir, img_name))
        return img_paths

    def get_lab_paths(self):
        lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            lab_paths.append(lab_path)
        return lab_paths

    def pad_and_scale(self, img, lab):
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(255, 255, 255))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(255, 255, 255))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize((self.img_size, self.img_size))
        padded_img = resize(padded_img)  # choose here

        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(lab, [0, 0, 0, pad_size], value=-1)
        else:
            padded_lab = lab
        return padded_lab


class SplitDataset:
    def __init__(self, img_dir, lab_dir, max_lab, img_size, transform=None):
        self.dataset = LisaDataset(img_dir=img_dir,
                                   lab_dir=lab_dir,
                                   max_lab=max_lab,
                                   img_size=img_size,
                                   transform=transform)
        self.img_dir = img_dir

    def __call__(self, val_split, test_split, shuffle_dataset, random_seed, batch_size, ordered, *args, **kwargs):
        if not ordered:
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))
            val_split = int(np.floor(val_split * dataset_size))
            test_split = int(np.floor(test_split * dataset_size))
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(indices)
            train_indices = indices[val_split + test_split:]
            val_indices = indices[:val_split]
            test_indices = indices[val_split:val_split + test_split]
        else:
            train_indices = []
            val_indices = []
            test_indices = []
            all_videos = self.get_frames_per_video()
            num_of_videos = len(all_videos)
            for idx, image in enumerate(os.listdir(self.img_dir)):
                video_name = image.split('.avi_')[0]
                if all_videos[video_name] < int((1 - val_split - test_split) * num_of_videos):
                    train_indices.append(idx)
                elif all_videos[video_name] < int((1 - val_split) * num_of_videos):
                    val_indices.append(idx)
                else:
                    test_indices.append(idx)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(self.dataset, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(self.dataset, sampler=test_sampler)

        return train_loader, validation_loader, test_loader

    def get_frames_per_video(self):
        videos = dict()
        idx = 1
        for image in os.listdir(self.img_dir):
            video_name = image.split('.avi_')[0]
            if video_name not in videos.keys():
                videos[video_name] = 1
                idx += 1
        indices_list = [i for i in range(idx)]
        for key, value in videos.items():
            rand_value = random.choice(indices_list)
            videos[key] = rand_value
            indices_list.remove(rand_value)
        return videos


class SplitDataset1:
    def __init__(self, img_dir_train_val, lab_dir_train_val, img_dir_test, lab_dir_test, max_lab, img_size, transform=None):
        self.dataset_train_val = LisaDataset(img_dir=img_dir_train_val,
                                             lab_dir=lab_dir_train_val,
                                             max_lab=max_lab,
                                             img_size=img_size,
                                             transform=transform)
        self.dataset_test = LisaDataset(img_dir=img_dir_test,
                                        lab_dir=lab_dir_test,
                                        max_lab=max_lab,
                                        img_size=img_size,
                                        transform=transform)
        self.img_dir = img_dir_train_val

    def __call__(self, val_split, test_split, shuffle_dataset, random_seed, batch_size, *args, **kwargs):
        train_indices = []
        val_indices = []
        all_videos = self.get_frames_per_video()
        num_of_videos = len(all_videos)
        for idx, image in enumerate(os.listdir(self.img_dir)):
            video_name = image.split('.avi_')[0]
            if all_videos[video_name] < int((1 - val_split) * num_of_videos):
                train_indices.append(idx)
            else:
                val_indices.append(idx)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(self.dataset_train_val, batch_size=batch_size, sampler=train_sampler)
        validation_loader = DataLoader(self.dataset_train_val, batch_size=batch_size, sampler=valid_sampler)
        test_loader = DataLoader(self.dataset_test)

        return train_loader, validation_loader, test_loader

    def get_frames_per_video(self):
        videos = dict()
        idx = 1
        for image in os.listdir(self.img_dir):
            video_name = image.split('.avi_')[0]
            if video_name not in videos.keys():
                videos[video_name] = 1
                idx += 1
        indices_list = [i for i in range(idx)]
        for key, value in videos.items():
            rand_value = random.choice(indices_list)
            videos[key] = rand_value
            indices_list.remove(rand_value)
        return videos


def main():
    img_dir = '../datasets/lisa/images'
    lab_dir = '../datasets/lisa/annotations'
    test_data_loader = torch.utils.data.DataLoader(LisaDataset(img_dir=img_dir,
                                                               lab_dir=lab_dir,
                                                               max_lab=14,
                                                               img_size=416),
                                                   batch_size=3,
                                                   shuffle=True)

    for i_batch, (img_batch, lab_batch) in enumerate(test_data_loader):
        for lab in lab_batch:
            print(lab)
        plt.imshow(img_batch[0].permute(1, 2, 0))
        plt.show()


if __name__ == '__main__':
    main()
