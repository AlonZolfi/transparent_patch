from PIL import Image
from torchvision import transforms
img = Image.open('..\datasets\lisa_detected\images\stop_1323817816.avi_image19.jpg')
patch = Image.open('C:/Users/alonz/Desktop/final_patch_w_alpha.png')

t_img = transforms.ToTensor()(transforms.Resize((608, 608))(img))
t_patch_w_alpha = transforms.ToTensor()(patch)
t_patch_wo_alpha = t_patch_w_alpha[:3]
alpha = t_patch_w_alpha[3]
merged = t_img * (1-alpha) + t_patch_wo_alpha * alpha

transforms.ToPILImage()(merged).save('try1.png')
