# from darknet import Darknet
# from utils import do_detect
from models.experimental import attempt_load
from detect import detect_boxes

import os
from PIL import Image
from shutil import copyfile

from torchvision import transforms

import torch

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect_stop_signs(yolo_gpu):
    dir_path = '../datasets/lisa/images/'
    dest_folder = '../datasets/lisa_yolov5m/'
    conf_threshold = 0.4
    cls_id = 11

    for image in os.listdir(dir_path):
        name = image.rsplit('.', maxsplit=1)[0]
        img_t = transforms.ToTensor()(transforms.Resize((608, 608))(Image.open(dir_path + image)))
        img_t_gpu = img_t.to(device)
        output = yolo_gpu(img_t_gpu.unsqueeze(0))[0]

        output = output.transpose(1, 2).contiguous()
        output_objectness, output = output[:, 4, :], output[:, 5:, :]

        confs_for_attacked_class = output[:, cls_id, :]
        conf = output_objectness * confs_for_attacked_class
        # find the max probability for stop sign
        max_conf, _ = torch.max(conf, dim=1)  # [batch]

        if max_conf.item() > conf_threshold:
            copyfile('../datasets/lisa/images/' + name + '.jpg', dest_folder + 'images/' + name + '.jpg')
            copyfile('../datasets/lisa/annotations/' + name + '.txt', dest_folder + 'annotations/' + name + '.txt')


def detect_other(model):
    dir_path = '../datasets/lisa_yolov5m/'
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
    conf_threshold = 0.4
    iou_threshold = 0.5

    for image in os.listdir(dir_path + 'images'):
        name = image.rsplit('.', maxsplit=1)[0]
        img = transforms.Resize((608, 608))(Image.open(dir_path + 'images/' + image))
        boxes = detect_boxes(model, img, classes, conf_threshold, iou_threshold)
        with open(dir_path + 'annotations/' + name + '.txt', 'r') as old_label:
            stop_sign_lines = ''.join(old_label.readlines())
        with open(dir_path + '/annotations_extra_labeled/' + name + '.txt', 'w') as text_file:
            text_file.write(stop_sign_lines)
            for box in boxes:
                cls_id = box[5]
                x_center = round(box[0], 3)
                y_center = round(box[1], 3)
                width = round(box[2], 3)
                height = round(box[3], 3)
                text_file.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')


def find_all_videos():
    s = dict()
    for image in os.listdir('../datasets/lisa_detected/images'):
        video_name = image.split('.avi_')[0]
        if video_name in s.keys():
            s[video_name] = s[video_name] + 1
        else:
            s[video_name] = 1
    print(s)


def detect():
    weight_file = '../yolov5-ultralytics/weights/yolov5m.pt'
    yolov5 = attempt_load(weight_file, map_location=device)
    # detect_stop_signs(yolov5)
    detect_other(yolov5)


detect()

