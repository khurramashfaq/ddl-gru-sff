import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import OpenEXR
import Imath
from PIL import Image

class Focus_on_Defocus(Dataset):
    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

        if self.mode == 'train':
            self.start_idx = 0
            self.end_idx = 399

        if self.mode == 'test':
            self.start_idx = 400
            self.end_idx = 499

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def load_data(self, idx):
        fs_num = self.start_idx + idx
        fs_num_str = f"{fs_num:06d}"

        exr = OpenEXR.InputFile(os.path.join(self.root_dir, fs_num_str + 'Dpt.exr'))
        dw = exr.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = {ch: np.frombuffer(exr.channel(ch, FLOAT), dtype=np.float32).reshape((height, width))
                    for ch in ['R', 'G', 'B']}

        dpt = np.stack([channels['R'], channels['G'], channels['B']], axis=-1)
        dpt_gray = np.average(dpt, 2)
        dpt_gray = torch.from_numpy(dpt_gray).unsqueeze(0).float()

        fs_images = []
        for j in range(5):
            img_name = fs_num_str + '_' + f'0{j}' + 'All.tif'
            img_path = os.path.join(self.root_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            fs_images.append(img)

        fs_images = torch.stack(fs_images)

        fs_images = (2 * fs_images - 1.0)


        return fs_images, dpt_gray

    def __getitem__(self, idx):
        return self.load_data(idx)





