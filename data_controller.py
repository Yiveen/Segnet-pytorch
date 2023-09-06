import torch
import numpy as np
from PIL import Image
import numpy.ma as ma
import torch.utils.data as data
import copy
from torchvision import transforms
import scipy.io as scio
import torchvision.datasets as dset
import random
import scipy.misc
import scipy.io as scio
import os
from PIL import ImageEnhance
from PIL import ImageFilter
import PIL

random.seed(2024)

class SegDataset(data.Dataset):
    def __init__(self, root_dir, txtlist, backtxt, use_noise):
        self.path = []
        self.real_path = []
        self.back = []
        self.use_noise = use_noise
        self.root = root_dir
        input_file = open(txtlist)
        backfile = open(backtxt)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
                self.path.append(copy.deepcopy(input_line))
            if input_line[5:].startswith("c"):
                self.real_path.append(copy.deepcopy(input_line))
        
        while 1:
            input_line = backfile.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
                self.back.append(copy.deepcopy(input_line))
            
        input_file.close()

        self.length = len(self.path)
        self.data_len = len(self.path)
        self.real_data_len = len(self.real_path)
        self.back_len = len(self.back)

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.back_front = np.array([[1 for i in range(640)] for j in range(480)])

    def __getitem__(self, idx):
        # index = random.randint(0, self.data_len - 10)

        label = np.array(Image.open('{0}/{1}/segmentation0000.png'.format(self.root, self.path[idx][:4])))
        if not self.use_noise:
            rgb = np.array(Image.open('{0}/{1}/Image0000.png'.format(self.root, self.path[idx][:4])).convert("RGB"))
        else:
            rgb = np.array(self.trancolor(Image.open('{0}/{1}/Image0000.png'.format(self.root, self.path[idx][:4])).convert("RGB")))

        if self.path[idx] in self.real_path:
            rgb = Image.open('{0}/{1}/Image0000.png'.format(self.root, self.path[idx][:4])).convert("RGB")
            rgb = ImageEnhance.Brightness(rgb).enhance(1.5).filter(ImageFilter.GaussianBlur(radius=0.8))
            rgb = np.array(self.trancolor(rgb))
            
            seed = random.randint(0, self.back_len-1)
            back = Image.open('background/{0}'.format(self.back[seed]))
            back = back.resize((640, 640), PIL.Image.LANCZOS)
            crop_size = (640,480)
            left = random.randint(0, 640 - crop_size[0])
            top = random.randint(0, 640 - crop_size[1])
            right = left + crop_size[0]
            bottom = top + crop_size[1]
            back = np.array(back.crop((left, top, right, bottom)))
   
            mask = ma.getmaskarray(ma.masked_equal(label, 0))[np.newaxis, :, :]
            back = np.transpose(back, (2, 0, 1))
            rgb = np.transpose(rgb, (2, 0, 1))

            rgb = back * mask + rgb * (1 - mask)
            # rgb = rgb.astype(np.uint8)
            # pil_rgb = Image.fromarray(rgb)
            # pil_rgb.save("rgb.png")
            # label = label.astype(np.uint8)
            # pil_mask = Image.fromarray(label)
            # pil_mask.save("mask.png")
            
            # Image.fromarray(rgb).save('embedding_final/rgb_{0}.png'.format(idx))
            # Image.fromarray(label).save('embedding_final/label_{0}.png'.format(idx))
            
            
        if self.use_noise:
            choice = random.randint(0, 3)
            if choice == 0:
                rgb = np.fliplr(rgb)
                label = np.fliplr(label)
            elif choice == 1:
                rgb = np.flipud(rgb)
                label = np.flipud(label)
            elif choice == 2:
                rgb = np.fliplr(rgb)
                rgb = np.flipud(rgb)
                label = np.fliplr(label)
                label = np.flipud(label)
                
        target = copy.deepcopy(label)
        if rgb.shape[0] != 3: 
            rgb = rgb.transpose(2, 0, 1)
        rgb = self.norm(torch.from_numpy(rgb.astype(np.float32)))
        target = torch.from_numpy(target.astype(np.int64))

        return rgb, target


    def __len__(self):
        return self.length

