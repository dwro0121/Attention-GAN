from torch.utils.data.dataset import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, img_folder_path, transform=None):
        self.img_folder_path = img_folder_path
        self.paths = os.listdir(self.img_folder_path)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(os.path.join(self.img_folder_path, img_path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)
