from utils import *
from darknet import Darknet

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def detect():
    cfg_file = 'cfg/yolo_v2-608.cfg'
    weight_file = 'weights/yolo_v2-608.weights'
    yolo = Darknet(cfg_file)
    yolo.load_weights(weight_file)
    yolo = yolo.to(device)

    for img_file in os.listdir('../datasets/lisa_detected/images'):
        img = Image.open('../datasets/lisa_detected/images/' + img_file).convert('RGB')
        sized = img.resize((yolo.width, yolo.height))
        boxes = do_detect(yolo, sized, 0.5, 0.4, True)
        with open('../datasets/extra_labeled/annotations_only_stop_sign/'+img_file.replace('.jpg', '.txt'), 'w') as f:
            for box in boxes:
                if box[6].item() != 11:
                    f.write(str(box[6].item()) + " "
                            + str(round(box[0].item(), 3)) + " "
                            + str(round(box[1].item(), 3)) + " "
                            + str(round(box[2].item(), 3)) + " "
                            + str(round(box[3].item(), 3)) + "\n")
            with open('../datasets/lisa_detected/annotations_only_stop_sign/' + img_file.replace('.jpg','.txt'), 'r') as true_file:
                lines = true_file.readlines()
                for line in lines:
                    f.write(line)

        # plot_boxes(img, boxes, 'predictions1.jpg', load_class_names('data/coco.names'))


detect()


