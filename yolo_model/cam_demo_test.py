from __future__ import division

import argparse
import pickle as pkl
import random
import time

from darknet import Darknet
from util import *


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.25)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="320", type=str)
    return parser.parse_args()


def get_files(model_type):
    if model_type == 'yolov3':
        cfg_file = "cfg/yolov3.cfg"
        weights_file = "weights/yolov3.weights"
        num_classes = 80
        classes_file = 'data/coco.names'
    elif model_type == 'yolov3-tiny':
        cfg_file = "cfg/yolov3-tiny.cfg"
        weights_file = "weights/yolov3-tiny.weights"
        num_classes = 80
        classes_file = 'data/coco.names'
    elif model_type == 'yolov3-yl-bulb':
        cfg_file = "cfg/yolov3-tl-small.cfg"
        weights_file = "weights/yolov3-tl-bulb.pth"
        num_classes = 2
        classes_file = 'data/tl-small.names'
    elif model_type == 'yolov3-tl-box':
        cfg_file = "cfg/yolov3-tl-small.cfg"
        weights_file = "weights/yolov3_ckpt_430.pth"
        num_classes = 2
        classes_file = 'data/tl-small.names'
    elif model_type == 'yolov3-signs':
        cfg_file = "cfg/trafficsigns-lisa.cfg"
        weights_file = "weights/trafficsigns-lisa_last.weights"
        num_classes = 47
        classes_file = 'data/trafficsigns-lisa.names'
    return cfg_file, weights_file, num_classes, classes_file


def get_cap(input_type):
    if input_type == 'video':
        videofile = 'videos/vid1trim1.mp4'
        return videofile
    elif input_type == 'local_camera':
        return 0
    elif input_type == 'ip_camera':
        return 'http://132.72.80.48:8081'


if __name__ == '__main__':
    model_type = 'yolov3'  # 'yolov3', 'yolov3-tiny', 'yolov3-tl-bulb', 'yolov3-tl-box', 'yolov3-signs'
    input_type = 'local_camera'  # 'video', 'local_camera', 'ip_camera'
    output_type = 'show'  # 'show', 'vid_out'

    cfg_file, weights_file, num_classes, classes_file = get_files(model_type)

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    print(CUDA)

    bbox_attrs = 5 + num_classes

    model = Darknet(cfg_file)
    if weights_file.endswith('weights'):
        model.load_weights(weights_file)
    else:
        model.load_state_dict(torch.load(weights_file))

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    cap = cv2.VideoCapture(get_cap(input_type))
    cap.set(3, 1280)
    cap.set(4, 720)

    if output_type == 'vid_out':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('out_videos/out1.avi', fourcc, 20.0, (640, 480))

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:

            img, orig_im, dim = prep_image(frame, inp_dim)

            if CUDA:
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thresh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

                if output_type == 'show':
                    cv2.imshow("frame", orig_im)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                else:
                    out.write(orig_im)

                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes(classes_file)
            colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, orig_im), output))

            if output_type == 'show':
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
            else:
                out.write(orig_im)

            frames += 1
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break
