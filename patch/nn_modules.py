import torch
from torch import nn
from torch.nn import functional as F

from torchvision import transforms

import numpy as np

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
        batch = (img_batch * (1.0-alpha)) + (adv_patch * alpha)
        # transforms.ToPILImage()(batch[0].cpu().squeeze(0)).show()
        # transforms.ToPILImage()(alpha.cpu().squeeze(0)).save('try2a.png')
        return batch


class DotApplier(nn.Module):
    def __init__(self, num_of_dots, img_size, alpha_max, beta_dropoff):
        super(DotApplier, self).__init__()
        self.img_size = img_size
        self.num_of_dots = num_of_dots
        self.alpha_max = alpha_max
        self.beta_dropoff = beta_dropoff
        self.zeros_like_circle = torch.zeros(size=(img_size, img_size), device=device)
        self.true_like_circle = torch.ones(size=(img_size, img_size), dtype=torch.bool, device=device)
        xx, yy = torch.from_numpy(np.mgrid[:img_size, :img_size])
        xx, yy = xx.to(device), yy.to(device)
        self.dist_from_circle_center = ((xx - img_size / 2) ** 2) + ((yy - img_size / 2) ** 2)

        # learnable parameters
        radius = torch.empty(size=(self.num_of_dots,), dtype=torch.float).uniform_(0.04, 0.09)
        self.radius = nn.Parameter(radius, requires_grad=True)

        colors = torch.empty(size=(self.num_of_dots, 3), dtype=torch.float).uniform_()
        self.colors = nn.Parameter(colors, requires_grad=True)

        theta = self.create_rand_translation()
        self.theta = nn.Parameter(theta, requires_grad=True)
        self.theta.register_hook(self.zero_grads)

    def get_circles(self, radius):
        circle = self.dist_from_circle_center / (radius * self.img_size) ** 2
        dist_from_circle_int = torch.where(circle > 1, self.zeros_like_circle, circle)
        dist_from_circle_bool = torch.where(circle <= 1, self.true_like_circle, ~self.true_like_circle)
        return dist_from_circle_bool, dist_from_circle_int

    def create_rand_translation(self):
        theta = torch.empty(size=(self.num_of_dots, 2, 3), dtype=torch.float)
        theta[:, 0, 0] = 1
        theta[:, 0, 1] = 0
        theta[:, 0, 2].uniform_(-0.9, 0.9)  # x axis
        theta[:, 1, 0] = 0
        theta[:, 1, 1] = 1
        theta[:, 1, 2].uniform_(0, 0.9)  # y axis
        return theta

    @staticmethod
    def zero_grads(grads):
        grads[:, 0, 0] = 0
        grads[:, 0, 1] = 0
        grads[:, 1, 0] = 0
        grads[:, 1, 1] = 0

    def forward(self, adv_patch, alpha):
        adv_patch.fill_(-1.)
        alpha.fill_(-1.)
        for i in range(self.num_of_dots):
            # draw circle in the middle of the image
            dot_tensor = self.draw_dot_on_image(i)
            # get affine matrix
            theta = self.theta[i][None]
            # get affine grid from affine matrix
            grid = F.affine_grid(theta, list(adv_patch.size()))
            # apply affine transformations - translation
            translated_dot = F.grid_sample(dot_tensor, grid, padding_mode='border')

            translated_dot_rgb = translated_dot[:, :3]
            adv_patch = torch.where((translated_dot_rgb >= 0), translated_dot_rgb, adv_patch)

            translated_alpha = translated_dot[:, -1]
            alpha = torch.where((translated_alpha >= 0), translated_alpha, alpha)

        adv_patch = torch.where((adv_patch < 0), torch.zeros_like(adv_patch), adv_patch)
        alpha = torch.where((alpha < 0), torch.zeros_like(alpha), alpha)
        # transforms.ToPILImage()(adv_patch.cpu().squeeze(0)).show()
        # transforms.ToPILImage()(alpha.cpu().squeeze(0)).show()
        return adv_patch, alpha

    def draw_dot_on_image(self, idx):
        dist_from_circle_bool, dist_from_circle_int = self.get_circles(self.radius[idx])
        color = self.colors[idx]
        r, g, b = color[0], color[1], color[2]

        blank_tensor = torch.full((1, 4, self.img_size, self.img_size), device=device, fill_value=-1.)
        blank_tensor[:, 0].masked_fill_(mask=dist_from_circle_bool, value=r)
        blank_tensor[:, 1].masked_fill_(mask=dist_from_circle_bool, value=g)
        blank_tensor[:, 2].masked_fill_(mask=dist_from_circle_bool, value=b)

        blank_tensor[:, 3].masked_fill_(mask=dist_from_circle_bool, value=self.alpha_max)
        blank_tensor[:, 3] *= ((-0.8*dist_from_circle_int**self.beta_dropoff)+1).squeeze(0)
        # blank_tensor[:, 3] *= torch.exp((-dist_from_circle_int)**self.beta_dropoff).squeeze(0)
        return blank_tensor


class WeightClipper(object):
    def __init__(self, radius_lower_bound, radius_upper_bound):
        self.radius_lower_bound = radius_lower_bound
        self.radius_upper_bound = radius_upper_bound

    def __call__(self, module):
        coordinates = module.theta[:, :, 2].data
        coordinates.clamp_(-0.99, 0.99)
        colors = module.colors.data
        colors.clamp_(0.001, 0.999)
        radius = module.radius.data
        radius.clamp_(self.radius_lower_bound, self.radius_upper_bound)


class DetectionsYolov5(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self, cls_id, num_cls, config, clean_img_conf, conf_threshold):
        super(DetectionsYolov5, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.clean_img_conf = clean_img_conf
        self.conf_threshold = conf_threshold

    def forward(self, lab_batch, output, img_names):
        def get_correct_ids():
            ids = np.unique(
                batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(int))
            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_conf[img_names[i]][lab_id])
            max_clean = np.array(clean_img_values)
            clean_gt = max_clean > self.conf_threshold
            return ids[clean_gt].tolist()

        output_patch = output.clone()
        output_patch = output_patch.transpose(1, 2).contiguous()

        output_objectness_patch, output_cls_patch = output_patch[:, 4, :], output_patch[:, 5:, :]

        batch_idx = lab_batch[..., 0:1]  # [(lab_batch[..., 0:1] != float(self.cls_id)) & (lab_batch[..., 0:1] != -1.)]
        total_loss = torch.empty(0).to(device)
        for i in range(batch_idx.size()[0]):
            ids = get_correct_ids()
            if len(ids) == 0:
                continue
            # get relevant classes
            confs_for_class_patch = output_cls_patch[i, ids, :]

            confs_if_object_patch = self.config.loss_target(output_objectness_patch[i][None], confs_for_class_patch)

            # find the max prob for each related class
            max_patch, _ = torch.max(confs_if_object_patch, dim=1)

            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_conf[img_names[i]][lab_id])

            # loss calc
            max_clean = torch.tensor(clean_img_values, device=device)
            curr_loss = torch.mean(torch.abs(max_clean-max_patch))  # get the mean of each image detections
            total_loss = torch.cat([total_loss, curr_loss.unsqueeze(0)])  # concat with prev results

        if total_loss.size()[0] == 0:
            total_loss = torch.zeros(1)

        confs_for_attacked_class = output_cls_patch[:, self.cls_id, :]  # [batch, 1805]
        confs_if_object = self.config.loss_target(output_objectness_patch, confs_for_attacked_class)  # [batch, 1805]
        # find the max probability for stop sign
        max_conf, _ = torch.max(confs_if_object, dim=1)  # [batch]
        return max_conf, total_loss


class NonPrintabilityScore(nn.Module):
    def __init__(self, printability_file, num_of_dots):
        super(NonPrintabilityScore, self).__init__()
        self.num_of_dots = num_of_dots
        self.printability_array = self.get_printability_array(printability_file)

    def get_printability_array(self, printability_file):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                str_arr = line.rstrip().split(",")
                float_arr = [float(num) for num in str_arr]
                printability_list.append(float_arr)

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full(self.num_of_dots, red))
            printability_imgs.append(np.full(self.num_of_dots, green))
            printability_imgs.append(np.full(self.num_of_dots, blue))
            printability_array.append(printability_imgs)

        printability_arr = torch.tensor(printability_array, dtype=torch.float32, device=device)
        return printability_arr

    def forward(self, dot_colors):
        # calculate RMSE distance between dot colors and printability array
        color_dist = (dot_colors.T - self.printability_array) + torch.finfo(torch.float32).eps
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + torch.finfo(torch.float32).eps
        color_dist = torch.sqrt(color_dist)
        # find the min distance between a dot color to printability array color
        color_dist_prod, _ = torch.min(color_dist, 0)
        # get the min between all colors
        nps_score = torch.mean(color_dist_prod)
        return nps_score


class NoiseAmount(nn.Module):
    def __init__(self, low, high):
        super(NoiseAmount, self).__init__()
        self.low = low
        self.high = high

    def forward(self, radiuses):
        noise = (radiuses - self.low) / (self.high - self.low)
        noise = noise.mean()
        return noise


class DetectionsYolov2(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """
    def __init__(self, weight_cls, weight_others, cls_id, num_cls, config, num_anchor, clean_img_conf, conf_threshold):
        super(DetectionsYolov2, self).__init__()
        self.weight_targeted = weight_cls
        self.weight_untargeted = weight_others
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config
        self.num_anchor = num_anchor
        self.clean_img_conf = clean_img_conf
        self.conf_threshold = conf_threshold

    def forward(self, lab_batch, output_patch, img_names):
        def get_correct_ids():
            ids = np.unique(
                batch_idx[i][(batch_idx[i] >= 0) & (batch_idx[i] != self.config.class_id)].cpu().numpy().astype(int))
            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_conf[img_names[i]][lab_id])
            max_clean = torch.tensor(clean_img_values, device=device)
            clean_gt = (max_clean > self.conf_threshold).cpu().numpy()
            return np.array(ids)[clean_gt].tolist()

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
            ids = get_correct_ids()
            if len(ids) == 0:
                continue
            # get relevant classes
            confs_for_class_patch = normal_confs_patch[i, ids, :]

            confs_if_object_patch = self.config.loss_target(output_objectness_patch[i][None], confs_for_class_patch)

            # find the max prob for each related class
            max_patch, _ = torch.max(confs_if_object_patch, dim=1)

            clean_img_values = []
            for lab_id in ids:
                clean_img_values.append(self.clean_img_conf[img_names[i]][lab_id])

            # loss calc
            max_clean = torch.tensor(clean_img_values, device=device)
            curr_loss = torch.mean(torch.abs(max_clean-max_patch))  # get the mean of each image detections
            total_loss = torch.cat([total_loss, curr_loss.unsqueeze(0)])  # concat with prev results

        if total_loss.size(0) == 0:
            total_loss = torch.zeros(1)

        confs_for_attacked_class = normal_confs_patch[:, self.cls_id, :]  # [batch, 1805]
        confs_if_object = self.config.loss_target(output_objectness_patch, confs_for_attacked_class)  # [batch, 1805]
        # find the max probability for stop sign
        max_conf, _ = torch.max(confs_if_object, dim=1)  # [batch]
        return self.weight_targeted * max_conf, self.weight_untargeted * total_loss

    def transform_output(self, output, batch, h, w):
        output = output.view(batch, self.num_anchor, 5 + self.num_cls, h * w)  # [batch, 5, 85, 361]
        output = output.transpose(1, 2).contiguous()  # [batch, 85, 5, 361]
        output = output.view(batch, 5 + self.num_cls, self.num_anchor * h * w)  # [batch, 85, 1805]
        output_objectness_patch = torch.sigmoid(output[:, 4, :])  # [batch, 1805]
        output = output[:, 5:5 + self.num_cls, :]  # [batch, 80, 1805]
        return output_objectness_patch, output


class DistanceDots(nn.Module):
    def __init__(self, weight, num_of_dots, img_size, radius):
        super(DistanceDots, self).__init__()
        self.weight = weight
        self.num_of_dots = num_of_dots
        self.img_size = img_size
        self.radius = radius
        self.max_colored_pixels = self.get_max_colored(num_of_dots)

    def get_max_colored(self, num_of_dots):
        true_like_circle = torch.ones(size=(self.img_size, self.img_size), dtype=torch.bool, device=device)
        false_like_circle = torch.zeros(size=(self.img_size, self.img_size), dtype=torch.bool, device=device)
        xx, yy = torch.from_numpy(np.mgrid[:self.img_size, :self.img_size])
        circle = (((xx - self.img_size / 2) ** 2) + ((yy - self.img_size / 2) ** 2)) / self.radius ** 2
        circle = circle.to(device)
        dist_from_circle_bool = torch.where(circle <= 1, true_like_circle, false_like_circle)
        return float(dist_from_circle_bool.sum().item() * num_of_dots)

    def forward(self, adv_patch):
        num_colored_patch = (adv_patch[:, 0] != 1).sum()
        colored_ratio = 1 - (num_colored_patch / self.max_colored_pixels)
        return colored_ratio * self.weight


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


class TotalVariation(nn.Module):
    def __init__(self, weight) -> None:
        super(TotalVariation, self).__init__()
        self.weight = weight

    def forward(self, adv_patch):
        h_tv = F.l1_loss(adv_patch[:, :, 1:, :], adv_patch[:, :, :-1, :], reduction='mean')  # calc height tv
        w_tv = F.l1_loss(adv_patch[:, :, :, 1:], adv_patch[:, :, :, :-1], reduction='mean')  # calc width tv
        loss = h_tv + w_tv
        return self.weight * loss

