from PIL import Image
from torchvision import transforms
img = Image.open('../datasets/bdd/images/train/6693767f-217403c5.jpg')
patch = Image.open('experiments/August/09-08-2020_20-15-04/final_results/final_patch_w_alpha.png')

t_img = transforms.ToTensor()(transforms.Resize((608, 608))(img))
t_patch_w_alpha = transforms.ToTensor()(patch)
t_patch_wo_alpha = t_patch_w_alpha[:3]
alpha = t_patch_w_alpha[3]
merged = t_img * (1-alpha) + t_patch_wo_alpha * alpha

transforms.ToPILImage()(merged).save('../yolov5-ultralytics/inference/images/try1.png')

