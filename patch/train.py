import torch

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
