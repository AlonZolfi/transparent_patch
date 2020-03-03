import torch
from torch import nn


class MaxProbExtractor(nn.Module):
    """
    MaxProbExtractor: extracts max class probability for class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, output):
        # get values necessary for transformation
        if output.dim() == 3:
            output = output.unsqueeze(0)

        batch = output.size(0)
        assert (output.size(1) == (5 + self.num_cls) * 5)

        h = output.size(2)
        w = output.size(3)

        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = output.view(batch, 5, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, 5 * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)

        # we only care for probabilities of the class of interest (person)
        confs_for_class = normal_confs[:, self.cls_id, :]

        # confs_if_object = output_objectness
        # confs_if_object = confs_for_class * output_objectness
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        # find the max probability for person
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)

        return max_conf


class PatchApplier(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch