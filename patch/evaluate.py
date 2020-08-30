from torchvision import transforms

from brambox.stat import pr, ap
from brambox.io import load

import os
import matplotlib.pyplot as plt
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import time
import torch
import numpy as np
from detect import detect_boxes


class EvaluateYOLO:
    def __init__(self, current_dir, test_loader, patch_applier, yolo, class_id, conf_threshold, iou_threshold, eval_data_type) -> None:
        self.current_dir = current_dir
        self.test_loader = test_loader
        self.patch_applier = patch_applier
        self.yolo = yolo
        self.class_id = class_id
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.base_path = 'testing/yolov5/medium/clean_lisa_' + eval_data_type + '/'

    def create_yolo_true_labels(self):
        Path(self.base_path + '/yolo-labels-target').mkdir(parents=True, exist_ok=True)
        Path(self.base_path + '/yolo-labels-other').mkdir(parents=True, exist_ok=True)
        save_dir_prefix = self.base_path + '/yolo-labels-'
        clean_results_coco_format_target = []
        clean_results_coco_format_other = []
        classes = np.arange(13).tolist()
        for img_batch, lab_batch, img_names in self.test_loader:
            name = os.path.splitext(img_names[0])[0]
            img = transforms.ToPILImage()(img_batch.cpu().squeeze(0))
            boxes = detect_boxes(self.yolo, img, classes, self.conf_threshold, self.iou_threshold)
            for box in boxes:
                cls_id = box[5]
                x_center = round(box[0], 3)
                y_center = round(box[1], 3)
                width = round(box[2], 3)
                height = round(box[3], 3)
                if cls_id == self.class_id:
                    with open(save_dir_prefix + 'target/' + name + '.txt', 'a') as text_file:
                        text_file.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                else:
                    with open(save_dir_prefix + 'other/' + name + '.txt', 'a') as text_file:
                        text_file.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            clean_result_coco_format_target, clean_result_coco_format_other = self.get_boxes_annotations(img, self.conf_threshold, self.iou_threshold, name)
            clean_results_coco_format_target.extend(clean_result_coco_format_target)
            clean_results_coco_format_other.extend(clean_result_coco_format_other)

        with open(self.base_path + 'target_clean_results.json', 'w') as fp:
            json.dump(clean_results_coco_format_target, fp)
        with open(self.base_path + 'other_clean_results.json', 'w') as fp:
            json.dump(clean_results_coco_format_other, fp)

    def plot_pr_curve(self, class_labels):
        cls_type = 'target' if len(class_labels) == 1 else 'other'
        class_label_map = self.get_class_label_map(class_labels)
        image_dims = {image_name.replace('.txt', ''): (1., 1.) for image_name in os.listdir(self.base_path + 'yolo-labels-' + cls_type + '/')}
        annotations = load(fmt='anno_darknet', path=self.base_path + 'yolo-labels-' + cls_type + '/', image_dims=image_dims, class_label_map=class_label_map)
        clean_results = load('det_coco', self.base_path + cls_type + '_clean_results.json', class_label_map=class_label_map)
        noise_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_noise_results.json', class_label_map=class_label_map)
        patch_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_patch_results.json', class_label_map=class_label_map)
        red_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_red_results.json', class_label_map=class_label_map)
        cyan_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_cyan_results.json', class_label_map=class_label_map)

        plt.figure()
        clean = pr(clean_results, annotations)
        clean_precision = clean['precision']
        clean_recall = clean['recall']
        noise = pr(noise_results, annotations)
        noise_precision = noise['precision']
        noise_recall = noise['recall']
        patch = pr(patch_results, annotations)
        patch_precision = patch['precision']
        patch_recall = patch['recall']
        red = pr(red_results, annotations)
        red_precision = red['precision']
        red_recall = red['recall']
        cyan = pr(cyan_results, annotations)
        cyan_precision = cyan['precision']
        cyan_recall = cyan['recall']

        plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
        title = 'Target Object' if cls_type == 'target' else 'Untargeted Objects'
        plt.title(title)
        ap_clean = ap(clean)
        plt.plot(clean_recall, clean_precision, label=f'CLEAN: AP: {round(ap_clean * 100, 2)}%')
        ap_noise = ap(noise)
        plt.plot(noise_recall, noise_precision, label=f'RANDOM: AP: {round(ap_noise * 100, 2)}%')
        ap_patch = ap(patch)
        plt.plot(patch_recall, patch_precision, label=f'PATCH: AP: {round(ap_patch * 100, 2)}%')
        ap_red = ap(red)
        plt.plot(red_recall, red_precision, label=f'RED: AP: {round(ap_red * 100, 2)}%')
        ap_cyan = ap(cyan)
        plt.plot(cyan_recall, cyan_precision, label=f'CYAN: AP: {round(ap_cyan * 100, 2)}%')

        plt.gca().set_ylabel('Precision')
        plt.gca().set_xlabel('Recall')
        plt.gca().set_xlim([0, 1.05])
        plt.gca().set_ylim([0, 1.05])

        handles, labels = plt.gca().get_legend_handles_labels()
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles),
                                      key=lambda t: float(t[0].split('AP: ')[1].replace('%', '')),
                                      reverse=True))
        plt.gca().legend(handles, labels, loc=4)

        plt.savefig(self.current_dir + '/final_results/' + cls_type + '-pr-curve.png')
        return ap_noise, ap_patch

    def create_jsons(self):
        print('Started creating evaluation jsons', flush=True)
        patch_file = self.current_dir + '/final_results/final_patch_wo_alpha.png'
        alpha_file = self.current_dir + '/final_results/alpha.png'
        rand_patch_file = self.current_dir + '/saved_patches/initial_patch.png'
        rand_alpha_file = self.current_dir + '/saved_patches/initial_alpha.png'
        adv_patch = transforms.ToTensor()(Image.open(patch_file))
        alpha = transforms.ToTensor()(Image.open(alpha_file))
        rand_adv_patch = transforms.ToTensor()(Image.open(rand_patch_file))
        rand_alpha = transforms.ToTensor()(Image.open(rand_alpha_file))
        red_adv_patch = torch.zeros_like(adv_patch)
        red_adv_patch[0].fill_(1)
        cyan_adv_patch = torch.ones_like(adv_patch)
        cyan_adv_patch[0].fill_(0)

        target_patch_results = []
        target_noise_results = []
        target_red_results = []
        target_cyan_results = []
        other_patch_results = []
        other_noise_results = []
        other_red_results = []
        other_cyan_results = []
        for img_batch, lab_batch, img_names in tqdm(self.test_loader):
            name = os.path.splitext(img_names[0])[0]

            # with patch
            applied_batch = self.patch_applier(img_batch, adv_patch, alpha)
            img = transforms.ToPILImage()(applied_batch.squeeze(0))
            target_patch_result, other_patch_result = self.get_boxes_annotations(img, 0.01, self.iou_threshold, name)
            target_patch_results.extend(target_patch_result)
            other_patch_results.extend(other_patch_result)

            # with rand patch
            rand_applied_batch = self.patch_applier(img_batch, rand_adv_patch, rand_alpha)
            img = transforms.ToPILImage()(rand_applied_batch.squeeze(0))
            target_noise_result, other_noise_result = self.get_boxes_annotations(img, 0.01, self.iou_threshold, name)
            target_noise_results.extend(target_noise_result)
            other_noise_results.extend(other_noise_result)

            # with red filter
            red_applied_batch = img_batch * red_adv_patch
            img = transforms.ToPILImage()(red_applied_batch.squeeze(0))
            target_red_result, other_red_result = self.get_boxes_annotations(img, 0.01, self.iou_threshold, name)
            target_red_results.extend(target_red_result)
            other_red_results.extend(other_red_result)

            # with cyan filter
            cyan_applied_batch = img_batch * cyan_adv_patch
            img = transforms.ToPILImage()(cyan_applied_batch.squeeze(0))
            target_cyan_result, other_cyan_result = self.get_boxes_annotations(img, 0.01, self.iou_threshold, name)
            target_cyan_results.extend(target_cyan_result)
            other_cyan_results.extend(other_cyan_result)

        with open(self.current_dir + '/testing/target_patch_results.json', 'w') as fp:
            json.dump(target_patch_results, fp)
        with open(self.current_dir + '/testing/target_noise_results.json', 'w') as fp:
            json.dump(target_noise_results, fp)
        with open(self.current_dir + '/testing/target_red_results.json', 'w') as fp:
            json.dump(target_red_results, fp)
        with open(self.current_dir + '/testing/target_cyan_results.json', 'w') as fp:
            json.dump(target_cyan_results, fp)
        with open(self.current_dir + '/testing/other_patch_results.json', 'w') as fp:
            json.dump(other_patch_results, fp)
        with open(self.current_dir + '/testing/other_noise_results.json', 'w') as fp:
            json.dump(other_noise_results, fp)
        with open(self.current_dir + '/testing/other_red_results.json', 'w') as fp:
            json.dump(other_red_results, fp)
        with open(self.current_dir + '/testing/other_cyan_results.json', 'w') as fp:
            json.dump(other_cyan_results, fp)
        print('Finished creating evaluation jsons', flush=True)

    def get_boxes_annotations(self, img, conf_threshold, nms_threshold, name):
        target_json_annotations = []
        other_json_annotations = []
        classes = np.arange(13).tolist()
        boxes = detect_boxes(self.yolo, img, classes, conf_threshold, nms_threshold)
        for box in boxes:
            cls_id = box[5]
            x_center = box[0]
            y_center = box[1]
            width = box[2]
            height = box[3]
            json_to_write = {'image_id': name,
                             'category_id': cls_id + 1,
                             'bbox': [round(x_center - (width / 2), 3),
                                      round(y_center - (height / 2), 3),
                                      round(width, 3),
                                      round(height, 3)],
                             'score': box[4]}
            if cls_id == self.class_id:
                target_json_annotations.append(json_to_write)
            else:
                other_json_annotations.append(json_to_write)
        return target_json_annotations, other_json_annotations

    def calculate(self):
        self.create_jsons()
        class_labels = [self.class_id]
        target_noise_ap, target_patch_ap = self.plot_pr_curve(class_labels)
        class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        other_noise_ap, other_patch_ap = self.plot_pr_curve(class_labels)
        return target_noise_ap, target_patch_ap, other_noise_ap, other_patch_ap

    @staticmethod
    def get_class_label_map(class_labels):
        label_map = dict()
        with open('coco.names') as classes_file:
            classes_lines = classes_file.readlines()
            for idx, line in enumerate(classes_lines):
                if idx in class_labels:
                    label_map[idx] = line.rstrip()
        return label_map