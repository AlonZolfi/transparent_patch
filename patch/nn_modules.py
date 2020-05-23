import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms
from torchvision.transforms import functional as TF

import numpy as np
from PIL import Image, ImageDraw

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatchApplier(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_patch, alpha):
        # adv_patch = adv_patch.permute(0, 2, 3, 1)
        alpha = torch.where((adv_patch == 1), torch.zeros(size=alpha.size(), device=device), alpha)
        batch = (img_batch * (1.0-alpha)) + (adv_patch * alpha)
        # transforms.ToPILImage()(batch[0].cpu()).show()
        # transforms.ToPILImage()(img_batch[0].cpu()).show()
        return batch


# class PreserveDetections(nn.Module):
#     """
#     PatchApplier: applies adversarial patches to images.
#     Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
#     """
#     def __init__(self, weight_cls, weight_others, cls_id, num_cls, config, num_anchor):
#         super(PreserveDetections, self).__init__()
#         self.weight_attack_cls = weight_cls
#         self.weight_others = weight_others
#         self.cls_id = cls_id
#         self.num_cls = num_cls
#         self.config = config
#         self.num_anchor = num_anchor
#
#     def forward(self, lab_batch, output_patch, output_clean):
#         batch_patch = output_patch.size(0)
#         batch_clean = output_clean.size(0)
#         assert (output_patch.size(1) == (5 + self.num_cls) * self.num_anchor)
#         h_patch = output_patch.size(2)
#         h_clean = output_clean.size(2)
#         w_patch = output_patch.size(3)
#         w_clean = output_clean.size(3)
#
#         # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
#         output_objectness_patch, output_patch = self.transform_output(output_patch, batch_patch, h_patch, w_patch)  # [batch, 1805]
#         output_objectness_clean, output_clean = self.transform_output(output_clean, batch_clean, h_clean, w_clean)  # [batch, 1805]
#
#         # perform softmax to normalize probabilities for object classes to [0,1]
#         normal_confs_patch = torch.nn.Softmax(dim=1)(output_patch)  # [batch, 80, 1805]
#         normal_confs_clean = torch.nn.Softmax(dim=1)(output_clean)  # [batch, 80, 1805]
#
#         batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
#         total_loss = torch.empty(0).to(device)
#         for i in range(batch_idx.size(0)):
#             ids = np.unique(batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.cls_id)].cpu().numpy().astype(int))
#             if len(ids) == 0:
#                 continue
#             # get relevant classes
#             confs_for_class_patch = normal_confs_patch[i, ids, :]
#             confs_for_class_clean = normal_confs_clean[i, ids, :]
#
#             confs_if_object_patch = self.config.loss_target(output_objectness_patch[i], confs_for_class_patch)
#             confs_if_object_clean = self.config.loss_target(output_objectness_clean[i], confs_for_class_clean)
#
#             # find the max prob for each related class
#             max_patch, _ = torch.max(confs_if_object_patch, dim=1)
#             max_clean, _ = torch.max(confs_if_object_clean, dim=1)
#
#             curr_loss = torch.mean(torch.abs(max_patch-max_clean))  # get the mean of each image detections
#
#             total_loss = torch.cat([total_loss, curr_loss.unsqueeze(0)])  # concat with prev results
#
#         if total_loss.size(0) == 0:
#             total_loss = torch.zeros(1)
#
#         confs_for_attacked_class = normal_confs_patch[:, self.cls_id, :]  # [batch, 1805]
#         confs_if_object = self.config.loss_target(output_objectness_patch, confs_for_attacked_class)  # [batch, 1805]
#         # find the max probability for stop sign
#         max_conf, _ = torch.max(confs_if_object, dim=1)  # [batch]
#
#         return self.weight_attack_cls * max_conf, self.weight_others * total_loss
#
#     def transform_output(self, output, batch, h, w):
#         output = output.view(batch, self.num_anchor, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
#         output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
#         output = output.view(batch, 5 + self.num_cls, self.num_anchor * h * w)  # [batch, 85, 1805]
#         output_objectness_patch = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
#         output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
#         return output_objectness_patch, output


class TotalVariation(nn.Module):
    def __init__(self, weight) -> None:
        super(TotalVariation, self).__init__()
        self.weight = weight

    def forward(self, adv_patch):
        h_tv = F.l1_loss(adv_patch[:, :, 1:, :], adv_patch[:, :, :-1, :], reduction='mean')  # calc height tv
        w_tv = F.l1_loss(adv_patch[:, :, :, 1:], adv_patch[:, :, :, :-1], reduction='mean')  # calc width tv
        loss = h_tv + w_tv
        return self.weight * loss


class DotApplier(nn.Module):
    def __init__(self, num_of_dots, img_size):
        super(DotApplier, self).__init__()
        self.img_size = img_size
        self.num_of_dots = num_of_dots
        self.theta = nn.Parameter(torch.tensor([[[1, 0, -0.5],
                                                [0, 1, -0.5]],
                                                [[1, 0, -0.3],
                                                 [0, 1, -0.4]]
                                                ], device=device), requires_grad=True)
        self.theta.register_hook(self.zero_grads)
        # stam = torch.tensor(-0.5)
        # self.theta = nn.Parameter(torch.tensor([[[1, 0, stam],
        #                                         [0, 1, -0.5]],
        #                                         [[1, 0, -0.7],
        #                                         [0, 1, -0.7]]
        #                                         ])[None], requires_grad=True)
        self.colors = [[50, 100, 150], [20, 215, 80]]

    def zero_grads(self, grads):
        grads[:, 0, 0] = 0
        grads[:, 0, 1] = 0
        grads[:, 1, 0] = 0
        grads[:, 1, 1] = 0

    def forward(self, adv_patch):
        for i in range(self.num_of_dots):
            dot_tensor = self.draw_dot_on_image(i)
            # transforms.ToPILImage()(dot_tensor.cpu().squeeze()).show()
            # get affine matrix
            theta = self.theta[i][None]
            # get affine grid from affine matrix
            grid = F.affine_grid(theta, list(adv_patch.size()))
            # apply affine transformations - translation
            translated_dot = F.grid_sample(dot_tensor, grid, padding_mode='border')
            # transforms.ToPILImage()(translated_dot.cpu().squeeze()).show()
            adv_patch = torch.where((translated_dot != 1), translated_dot, adv_patch)
            transforms.ToPILImage()(adv_patch.cpu().squeeze()).show()
        return adv_patch

    def draw_dot_on_image(self, idx):
        mask_img = Image.new('RGB', size=(self.img_size, self.img_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(mask_img)
        x_center = 51
        y_center = 51
        radius = 50
        draw.ellipse((x_center - radius, y_center - radius, x_center + radius, y_center + radius), fill=(0, 0, 0))
        one_indices = (transforms.ToTensor()(mask_img) == 0)
        color = self.colors[idx]
        r, g, b = color[0] / 255, color[1] / 255, color[2] / 255
        blank_tensor = torch.full((1, 3, self.img_size, self.img_size), dtype=torch.float32, fill_value=1)
        blank_tensor[:, 0].masked_fill_(mask=one_indices[0], value=r)
        blank_tensor[:, 1].masked_fill_(mask=one_indices[1], value=g)
        blank_tensor[:, 2].masked_fill_(mask=one_indices[2], value=b)
        return blank_tensor.to(device)


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


class PreserveDetections(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self, weight_cls, weight_others, cls_id, num_cls, config, num_anchor, clean_img_conf):
        super(PreserveDetections, self).__init__()
        self.weight_attack_cls = weight_cls
        self.weight_others = weight_others
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.num_anchor = num_anchor
        self.clean_img_conf = clean_img_conf

    def forward(self, lab_batch, output_patch, img_name):
        batch_patch = output_patch.size(0)
        assert (output_patch.size(1) == (5 + self.num_cls) * self.num_anchor)
        h_patch = output_patch.size(2)
        w_patch = output_patch.size(3)

        # transform the output tensor from [batch, 425, 19, 19] to [batch, 80, 1805]
        output_objectness_patch, output_patch = self.transform_output(output_patch, batch_patch, h_patch, w_patch)  # [batch, 1805]

        # perform softmax to normalize probabilities for object classes to [0,1]
        normal_confs_patch = torch.nn.Softmax(dim=1)(output_patch)  # [batch, 80, 1805]

        batch_idx = torch.index_select(lab_batch, 2, torch.tensor([0], dtype=torch.long).to(device))
        total_loss = torch.empty(0).to(device)
        for i in range(batch_idx.size(0)):
            ids = np.unique(batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.cls_id)].cpu().numpy().astype(int))
            if len(ids) == 0:
                continue
            # get relevant classes
            confs_for_class_patch = normal_confs_patch[i, ids, :]

            confs_if_object_patch = self.config.loss_target(output_objectness_patch[i], confs_for_class_patch)

            # find the max prob for each related class
            max_patch, _ = torch.max(confs_if_object_patch, dim=1)

            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_conf[img_name[i]][lab_id])
            max_clean = torch.tensor(clean_img_values, device=device)

            curr_loss = torch.mean(torch.abs(max_clean-max_patch))  # get the mean of each image detections

            total_loss = torch.cat([total_loss, curr_loss.unsqueeze(0)])  # concat with prev results

        if total_loss.size(0) == 0:
            total_loss = torch.zeros(1)

        confs_for_attacked_class = normal_confs_patch[:, self.cls_id, :]  # [batch, 1805]
        confs_if_object = self.config.loss_target(output_objectness_patch, confs_for_attacked_class)  # [batch, 1805]
        # find the max probability for stop sign
        max_conf, _ = torch.max(confs_if_object, dim=1)  # [batch]

        return self.weight_attack_cls * max_conf, self.weight_others * total_loss

    def transform_output(self, output, batch, h, w):
        output = output.view(batch, self.num_anchor, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, self.num_anchor * h * w)  # [batch, 85, 1805]
        output_objectness_patch = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
        return output_objectness_patch, output
