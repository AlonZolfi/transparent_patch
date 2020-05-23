from torch import optim


class BaseConfiguration:
    def __init__(self):
        """
        Set the defaults.
        """
        self.patch_name = 'base'
        self.img_dir = '../datasets/lisa_debug/images'
        self.lab_dir = '../datasets/lisa_debug/annotations'
        self.cfg_file = '../pytorch-yolo2/cfg/yolo_v2-608.cfg'
        self.weight_file = '../pytorch-yolo2/weights/yolov2-608.weights'
        # self.printfile = '/home/zolfi/adversarial_yolo_patch/attack/non_printability/30values.txt'

        self.num_classes = 80
        self.class_id = 11
        self.patch_size = 608
        self.max_labels_per_img = 40

        self.start_learning_rate = 0.01
        self.epochs = 3
        self.batch_size = 4

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=3)
        self.loss_target = lambda obj, cls: obj * cls

        self.max_tv = 0.1
        self.tv_weight = 6
        self.max_prob_weight = 3
        self.pres_det_weight = 1

        self.num_of_dots = 2
        # self.alpha = 0.1


class TrainingOnCluster(BaseConfiguration):
    def __init__(self):
        super(TrainingOnCluster, self).__init__()
        self.patch_name = 'cluster'
        self.batch_size = 16


class TrainingOnPrivateComputer(BaseConfiguration):
    def __init__(self):
        super(TrainingOnPrivateComputer, self).__init__()
        self.patch_name = 'private'
        self.batch_size = 2


patch_config_types = {
    "base": BaseConfiguration,
    "cluster": TrainingOnCluster,
    "private": TrainingOnPrivateComputer
}
