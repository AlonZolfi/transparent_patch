from torch import optim


class BaseConfiguration:
    def __init__(self):
        self.patch_name = 'base'
        self.img_dir = '../datasets/lisa_new/images'
        self.lab_dir = '../datasets/lisa_new/annotations_extra_labeled'
        self.cfg_file = '../pytorch-yolo2/cfg/yolo_v2-608.cfg'
        self.weight_file = '../pytorch-yolo2/weights/yolo_v2-608.weights'
        self.print_file = '../patch/30values.txt'

        self.num_classes = 80
        self.class_id = 11
        self.patch_size = 608
        self.max_labels_per_img = 40

        self.epochs = 100
        self.batch_size = 4
        self.conf_threshold = 0.5

        self.sched_cooldown = 0
        self.sched_patience = 2
        self.loc_lr = 0.007
        self.color_lr = 0.007
        self.radius_lr = 0.007
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        mode='min',
                                                                                        cooldown=self.sched_cooldown,
                                                                                        patience=self.sched_patience,
                                                                                        factor=0.9,
                                                                                        min_lr=0.001)
        self.loss_mode = 'cls'  # 'obj * cls', 'obj'
        self.loss_target = lambda obj, cls: self.get_loss(self.loss_mode, obj, cls)

        self.max_prob_weight = 1
        self.pres_det_weight = 0
        self.nps_weight = 0

        self.num_of_dots = 10
        self.alpha_max = 0.4
        self.beta_dropoff = 2

    def get_loss(self, loss_mode, obj, cls):
        if loss_mode == 'obj * cls':
            return obj * cls
        elif loss_mode == 'obj':
            return obj
        else:
            return cls


class TrainingOnCluster(BaseConfiguration):
    def __init__(self):
        super(TrainingOnCluster, self).__init__()
        self.patch_name = 'cluster'
        self.batch_size = 8


class TrainingOnPrivateComputer(BaseConfiguration):
    def __init__(self):
        super(TrainingOnPrivateComputer, self).__init__()
        self.patch_name = 'private'
        self.batch_size = 4


patch_config_types = {
    "base": BaseConfiguration,
    "cluster": TrainingOnCluster,
    "private": TrainingOnPrivateComputer
}
