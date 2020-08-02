from darknet import Darknet
from utils import do_detect

import os
from PIL import Image
from shutil import copyfile

from torchvision import transforms

import torch

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect_stop_signs():
    cfg_file = '../pytorch-yolo2/cfg/yolo_v2-608.cfg'
    weight_file = '../pytorch-yolo2/weights/yolo_v2-608.weights'
    dir_path = '../datasets/lisa/images/'

    yolo = Darknet(cfg_file)
    yolo.load_weights(weight_file)
    yolo.eval()
    yolo_gpu = yolo.to(device)

    conf_threshold = 0.5
    num_cls = 80
    cls_id = 11

    for image in os.listdir(dir_path):
        name = image.rsplit('.', maxsplit=1)[0]
        img_t = transforms.ToTensor()(transforms.Resize((608, 608))(Image.open(dir_path+image)))
        img_t_gpu = img_t.to(device)
        output = yolo_gpu(img_t_gpu.unsqueeze(0))

        h = output.size(2)
        w = output.size(3)

        output = output.view(1, yolo.num_anchors, 5 + num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(1, 5 + num_cls, yolo.num_anchors * h * w)  # [batch, 85, 1805]
        output_objectness_patch = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + num_cls, :]  # [batch, 80, 1805]
        normal_confs_patch = torch.nn.Softmax(dim=1)(output)

        confs_for_attacked_class = normal_confs_patch[:, cls_id, :]
        # confs_if_object = output_objectness_patch * confs_for_attacked_class  # [batch, 1805]
        # find the max probability for stop sign
        max_conf_objectness, _ = torch.max(output_objectness_patch, dim=1)  # [batch]
        max_conf_class, _ = torch.max(confs_for_attacked_class, dim=1)  # [batch]

        if max_conf_objectness.item() > conf_threshold and max_conf_class > conf_threshold:
            copyfile('../datasets/lisa/images/'+name+'.jpg', '../datasets/lisa_new/images/'+name+'.jpg')
            copyfile('../datasets/lisa/annotations/'+name+'.txt', '../datasets/lisa_new/annotations/'+name+'.txt')


def detect_other():
    cfg_file = '../pytorch-yolo2/cfg/yolo_v2-608.cfg'
    weight_file = '../pytorch-yolo2/weights/yolo_v2-608.weights'
    dir_path = '../datasets/lisa_new/images/'

    yolo = Darknet(cfg_file)
    yolo.load_weights(weight_file)
    yolo.eval()
    yolo_gpu = yolo.to(device)

    conf_threshold = 0.5
    nms_threshold = 0.4
    class_labels = [0,1,2,3,4,5,6,7,8,9,10,12]

    for image in os.listdir(dir_path):
        name = image.rsplit('.', maxsplit=1)[0]
        img = transforms.Resize((608, 608))(Image.open(dir_path+image))
        boxes = do_detect(yolo_gpu, img, conf_threshold, nms_threshold, True)
        with open('../datasets/lisa_new/annotations/' + name + '.txt', 'r') as old_label:
            stop_sign_lines = ''.join(old_label.readlines())
        with open('../datasets/lisa_new/annotations_extra_labeled/' + name + '.txt', 'w') as text_file:
            text_file.write(stop_sign_lines)
            for box in boxes:
                cls_id = box[6].item()
                if cls_id in class_labels:
                    x_center = round(box[0].item(), 3)
                    y_center = round(box[1].item(), 3)
                    width = round(box[2].item(), 3)
                    height = round(box[3].item(), 3)
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


# find_all_videos()

# detect_stop_signs()
detect_other()