import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import OpenEXR
import Imath
import torchvision.transforms as transforms

class FT(Dataset):
    def __init__(self, root_dir, mode='train',crop_train=True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.crop_train= crop_train
        self.original_size = (540, 960)
        self.input_size = (544, 960)
        self.resize_transform = transforms.Resize(self.input_size)

        if self.mode == 'train':
            self.data_dir = os.path.join(self.root_dir, 'train', 'focal_stack')
        elif self.mode == 'test':
            self.data_dir = os.path.join(self.root_dir, 'val', 'focal_stack')
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        self.folders = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.folders)

    def read_exr(self, exr_path):
        exr = OpenEXR.InputFile(exr_path)
        dw = exr.header()['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        disp_data = np.frombuffer(exr.channel('Y', FLOAT), dtype=np.float32)
        disp = np.reshape(disp_data, (height, width))
        return disp

    def __getitem__(self, idx):
        folder_path = os.path.join(self.data_dir, self.folders[idx])


        focal_stack_images = []
        img_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.png') and 'AiF' not in f],
            key=lambda fn: float(os.path.splitext(fn)[0])
        )

        for img_file in img_files:
            img = Image.open(os.path.join(folder_path, img_file)).convert('RGB')
            focal_stack_images.append(np.array(img))


        aif_path = os.path.join(folder_path, 'AiF.png')
        aif_image = np.array(Image.open(aif_path).convert('RGB'))


        disp_path = os.path.join(folder_path, 'disp.exr')
        disparity = self.read_exr(disp_path)

        if self.mode == 'test':
            padded_stack = []
            for img in focal_stack_images:
                padded_stack.append(np.pad(img, ((2, 2), (0, 0), (0, 0)), mode='edge'))
            focal_stack_images = np.array(padded_stack)
            aif_image = np.pad(aif_image, ((2, 2), (0, 0), (0, 0)), mode='edge')
            disparity = np.pad(disparity, ((2, 2), (0, 0)), mode='edge')
        else:
            focal_stack_images = np.array(focal_stack_images)
            if self.crop_train:
              H, W = focal_stack_images.shape[1:3]
              top = np.random.randint(0, H - 256)
              left = np.random.randint(0, W - 256)
              focal_stack_images = focal_stack_images[:, top:top + 256, left:left + 256, :]
              aif_image = aif_image[top:top + 256, left:left + 256, :]
              disparity = disparity[top:top + 256, left:left + 256]


        focal_stack = torch.from_numpy(focal_stack_images).permute(0, 3, 1, 2).float()
        aif_tensor = torch.from_numpy(aif_image).permute(2, 0, 1).float()
        disparity = torch.from_numpy(disparity.copy()).unsqueeze(0).float()

        focal_stack_images = (2 * (focal_stack / 255)) - 1
        aif_image = (2 * (aif_tensor / 255)) - 1


        return focal_stack_images, aif_image, disparity


