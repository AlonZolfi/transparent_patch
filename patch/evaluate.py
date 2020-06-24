from torchvision import transforms

from brambox.stat import pr, ap
from brambox.io import load

import os
import matplotlib.pyplot as plt
import json
from PIL import Image

from utils import do_detect


class EvaluateYOLO:
    def __init__(self, current_dir, test_loader, patch_applier, yolo, class_id, conf_threshold) -> None:
        self.current_dir = current_dir
        self.test_loader = test_loader
        self.patch_applier = patch_applier
        self.yolo = yolo
        self.class_id = class_id
        self.conf_threshold = conf_threshold

    def create_yolo_true_labels(self, class_labels):
        cls_type = 'target' if len(class_labels) == 1 else 'other'
        save_dir = 'testing/clean/yolo-labels-' + cls_type + '/'
        clean_results_coco_format = []
        for img_batch, lab_batch, img_names in self.test_loader:
            name = os.path.splitext(img_names[0])[0]
            img = transforms.ToPILImage()(img_batch.cpu().squeeze(0))
            boxes = do_detect(self.yolo, img, self.conf_threshold, 0.4, True)
            with open(save_dir + name + '.txt', 'w+') as text_file:
                for box in boxes:
                    cls_id = box[6].item()
                    if cls_id in class_labels:
                        x_center = round(box[0].item(), 3)
                        y_center = round(box[1].item(), 3)
                        width = round(box[2].item(), 3)
                        height = round(box[3].item(), 3)
                        text_file.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            clean_results_coco_format.extend(self.get_boxes_annotations(img, self.conf_threshold, 0.4, name, class_labels))

        with open('testing/clean/' + cls_type + '_clean_results.json', 'w') as fp:
            json.dump(clean_results_coco_format, fp)

    def plot_pr_curve(self, class_labels):
        cls_type = 'target' if len(class_labels) == 1 else 'other'
        class_label_map = self.get_class_label_map(class_labels)
        image_dims = {image_name.replace('.txt', ''): (1., 1.) for image_name in os.listdir('testing/clean/yolo-labels-' + cls_type + '/')}
        annotations = load(fmt='anno_darknet', path='testing/clean/yolo-labels-' + cls_type + '/', image_dims=image_dims, class_label_map=class_label_map)
        clean_results = load('det_coco', 'testing/clean/' + cls_type + '_clean_results.json', class_label_map=class_label_map)
        noise_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_rand_noise_results.json', class_label_map=class_label_map)
        patch_results = load('det_coco', self.current_dir + '/testing/' + cls_type + '_patch_results.json', class_label_map=class_label_map)

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

        plt.plot([0, 1.05], [0, 1.05], '--', color='gray')
        title = 'Target Object' if cls_type == 'target' else 'Other Objects'
        plt.title(title)
        ap_clean = ap(clean)
        plt.plot(clean_recall, clean_precision, label=f'CLEAN: AP: {round(ap_clean * 100, 2)}%')
        ap_noise = ap(noise)
        plt.plot(noise_recall, noise_precision, label=f'NOISE: AP: {round(ap_noise * 100, 2)}%')
        ap_patch = ap(patch)
        plt.plot(patch_recall, patch_precision, label=f'PATCH: AP: {round(ap_patch * 100, 2)}%')
        plt.gca().set_ylabel('Precision')
        plt.gca().set_xlabel('Recall')
        plt.gca().set_xlim([0, 1.05])
        plt.gca().set_ylim([0, 1.05])
        plt.gca().legend(loc=4)
        plt.savefig(self.current_dir + '/final_results/' + cls_type + '-pr-curve.png')
        # plt.show()
        # self.test_ap = ap_patch

    def create_jsons(self, class_labels):
        print('Started creating evaluation jsons')
        cls_type = 'target' if len(class_labels) == 1 else 'other'
        patch_file = self.current_dir + '/final_results/final_patch_wo_alpha.png'
        alpha_file = self.current_dir + '/final_results/alpha.png'
        rand_patch_file = self.current_dir + '/saved_patches/initial_patch.png'
        rand_alpha_file = self.current_dir + '/saved_patches/initial_alpha.png'
        adv_patch = transforms.ToTensor()(Image.open(patch_file))
        alpha = transforms.ToTensor()(Image.open(alpha_file))
        rand_adv_patch = transforms.ToTensor()(Image.open(rand_patch_file))
        rand_alpha = transforms.ToTensor()(Image.open(rand_alpha_file))

        patch_results = []
        rand_noise_results = []
        for img_batch, lab_batch, img_names in self.test_loader:
            name = os.path.splitext(img_names[0])[0]

            # with patch
            applied_batch = self.patch_applier(img_batch, adv_patch, alpha)
            img = transforms.ToPILImage()(applied_batch.squeeze(0))
            patch_results.extend(self.get_boxes_annotations(img, 0.0001, 0.4, name, class_labels))

            # with rand patch
            rand_applied_batch = self.patch_applier(img_batch, rand_adv_patch, rand_alpha)
            img = transforms.ToPILImage()(rand_applied_batch.squeeze(0))
            rand_noise_results.extend(self.get_boxes_annotations(img, 0.0001, 0.4, name, class_labels))

        with open(self.current_dir + '/testing/' + cls_type + '_patch_results.json', 'w') as fp:
            json.dump(patch_results, fp)
        with open(self.current_dir + '/testing/' + cls_type + '_rand_noise_results.json', 'w') as fp:
            json.dump(rand_noise_results, fp)

        print('Finished creating evaluation jsons')

    def get_boxes_annotations(self, img, conf_threshold, nms_threshold, name, class_labels):
        json_annotations = []
        boxes = do_detect(self.yolo, img, conf_threshold, nms_threshold, True)
        for box in boxes:
            cls_id = box[6].item()
            if cls_id in class_labels:
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                json_annotations.append({'image_id': name,
                                         'bbox': [round(x_center.item() - width.item() / 2, 3),
                                                  round(y_center.item() - height.item() / 2, 3),
                                                  round(width.item(), 3),
                                                  round(height.item(), 3)],
                                         'score': box[4].item(),
                                         'category_id': cls_id + 1})
        return json_annotations

    def calculate(self):
        class_labels = [self.class_id]
        self.create_jsons(class_labels)
        self.plot_pr_curve(class_labels)
        class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
        self.create_jsons(class_labels)
        self.plot_pr_curve(class_labels)

    def get_class_label_map(self, class_labels):
        label_map = dict()
        with open('../pytorch-yolo2/data/coco.names') as classes_file:
            classes_lines = classes_file.readlines()
            for idx, line in enumerate(classes_lines):
                if idx in class_labels:
                    label_map[idx] = line.rstrip()
        return label_map

