import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision

class FlowersDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.class_to_idx = {class_name: i for i, class_name in enumerate(os.listdir(data_path))}
        self.samples = [(os.path.join(data_path, class_name, image), self.class_to_idx[class_name]) for class_name in os.listdir(data_path) for image in os.listdir(os.path.join(data_path, class_name))]
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img = Image.open(self.samples[index][0])
        img = torchvision.transforms.Resize((224,224))(img)
        img = torchvision.transforms.PILToTensor()(img).float()
        return (img, torch.Tensor(self.samples[index][1]).long())
        
    


