from torch import optim


class BaseConfiguration:
    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = '../datasets/lisa/images'
        self.lab_dir = '../datasets/lisa/annotations'
        self.cfg_file = "../pytorch-yolo2/cfg/yolo_v2-608.cfg"
        self.weight_file = "../pytorch-yolo2/weights/yolov2-608.weights"
        #self.printfile = "/home/zolfi/adversarial_yolo_patch/attack/non_printability/30values.txt"
        self.num_classes = 80
        self.class_id = 11
        self.patch_size = 608

        self.start_learning_rate = 0.03
        self.epochs = 1000
        self.batch_size = 4

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.loss_target = lambda obj, cls: obj * cls

        self.max_tv = 0.01
        self.tv_weight = 2.5
        self.max_prob_weight = 0.1
        self.iou_weight = 0.1

        self.num_of_dots = 10
        self.alpha = 0.2


patch_config_types = {
    "base": BaseConfiguration
}