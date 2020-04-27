import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms

from PIL import ImageDraw, Image


class MaxProbExtractor(nn.Module):
    """
    MaxProbExtractor: extracts max class probability for class from YOLO output.
    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.
    """

    def __init__(self, weight, cls_id, num_cls, config, num_anchor):
        super(MaxProbExtractor, self).__init__()
        self.weight = weight
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.num_anchor = num_anchor

    def forward(self, output):
        batch = output.size(0)
        assert (output.size(1) == (5 + self.num_cls) * self.num_anchor)
        h = output.size(2)
        w = output.size(3)

        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output = output.view(batch, self.num_anchor, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, self.num_anchor * h * w)  # [batch, 85, 1805]
        output_objectness = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        bboxes = output[:, :4, :]  # [batch, 4, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs = torch.nn.Softmax(dim=1)(output)  # [batch, 80, 1805]

        # we only care for probabilities of the class of interest (stop sign)
        confs_for_class = normal_confs[:, self.cls_id, :]  # [batch, 1805]

        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)  # [batch, 1805]

        # find the max probability for stop sign
        max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)  # [batch]
        max_bboxes = torch.index_select(bboxes, dim=2, index=max_conf_idx)

        return self.weight * max_conf


class PatchApplier(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_patch, alpha):
        img_batch = (img_batch * (1.0-alpha)) + (adv_patch * alpha)
        return img_batch


class PatchTrainer(nn.Module):
    def __init__(self, num_of_dots, img_size) -> None:
        super(PatchTrainer, self).__init__()
        self.img_size = img_size
        self.num_of_dots = num_of_dots
        self.param_list = nn.ParameterList()
        for i in range(num_of_dots):
            center_location = nn.Parameter(torch.rand(2), requires_grad=True)
            radius = nn.Parameter(torch.rand(1), requires_grad=True)
            color = nn.Parameter(torch.rand(3), requires_grad=True)

            self.param_list.append(center_location)
            self.param_list.append(radius)
            self.param_list.append(color)

    def forward(self, adv_patch):
        for i in range(self.num_of_dots):
            mask_img = Image.new('RGB', size=(self.img_size, self.img_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(mask_img)

            center = self.state_dict().get('param_list.'+str(3*i))
            radius = self.state_dict().get('param_list.'+str((3*i)+1))
            tensor_color = self.state_dict().get('param_list.'+str((3*i)+2))
            color = [c.item() for c in tensor_color]
            r, g, b = color[0], color[1], color[2]
            x_center, y_center, radius = self.get_correct_param_values(center, radius)

            draw.ellipse((x_center-radius, y_center-radius, x_center+radius, y_center+radius), fill=(0, 0, 0))
            one_indices = (transforms.ToTensor()(mask_img) == 0)

            adv_patch[0].masked_fill_(mask=one_indices[0], value=r)
            adv_patch[1].masked_fill_(mask=one_indices[1], value=g)
            adv_patch[2].masked_fill_(mask=one_indices[2], value=b)

        return adv_patch

    def get_correct_param_values(self, center, radius):
        x_center = center[0] * self.img_size
        y_center = center[1] * self.img_size
        radius = radius * (self.img_size / 5)
        return int(x_center), int(y_center), int(radius)


class TotalVariation(nn.Module):
    def __init__(self, weight) -> None:
        super(TotalVariation, self).__init__()
        self.weight = weight

    def forward(self, adv_patch):
        h_tv = F.l1_loss(adv_patch[:, 1:, :], adv_patch[:, :-1, :], reduction='mean')  # calc height tv
        w_tv = F.l1_loss(adv_patch[:, :, 1:], adv_patch[:, :, :-1], reduction='mean')  # calc width tv
        loss = h_tv + w_tv
        return self.weight * loss


class IoU(nn.Module):
    def __init__(self, weight) -> None:
        super(IoU, self).__init__()
        self.weight = weight

    def forward(self, bbox1, bbox2):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
            bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        inter = self.intersect(bbox1, bbox2)
        area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                  (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                  (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

