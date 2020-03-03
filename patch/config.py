from torch import optim


class BaseConfiguration:
    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = '../datasets/lisa/images'
        self.lab_dir = '../datasets/lisa/annotations'
        self.cfg_file = "../yolo_model/cfg/yolov3.cfg"
        self.weight_file = "../yolo_model/weights/yolov3.weights"
        #self.printfile = "/home/zolfi/adversarial_yolo_patch/attack/non_printability/30values.txt"
        self.num_classes = 80
        self.class_id = 11
        self.patch_size = 300

        self.start_learning_rate = 0.03
        self.epochs = 1000

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.loss_target = lambda obj, cls: obj * cls


patch_config_types = {
    "base": BaseConfiguration
}