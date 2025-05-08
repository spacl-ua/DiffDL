import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
#PM Edits
import random
import re
#PM End Edits

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
	# PM_Edits
	# images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
	# END_PM Edits
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')
#PM Edits
def pil_gray_loader(path):
    return Image.open(path).convert('L').convert('RGB')
#END_PM Edits

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
	#PM Edit
	self.gray_loader = gray_loader
	#End_PM Edit
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'
	
	#PM Edit
	path = self.flist[index]
	#End_PM Edit
        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

	#PM Edit
	#img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
	#cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))
	#End PM Edit

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
	#PM Edit
	#ret['path'] = file_name
	filename = os.path.split(path)
        ret['path'] = file_name[1] #<- added [1]
	#End Edit
        return ret

    def __len__(self):
        return len(self.flist)

#PM Edit
def numpy_loader(path):
    return np.load(path)
#End Edit

#PM_Edit
class DiffusionDKIDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, targetMetric='KFA', numInputDWIs=6, image_size=[192, 192], loader=numpy_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = numpy_loader
        self.image_size = image_size
        self.targetMetric = targetMetric
        self.numInputDWIs = numInputDWIs

    def __getitem__(self, index):
        ret = {}
        path=self.data_root+self.flist[index]

        data = self.loader(path)

        # Prep input img
        if self.targetMetric=='AK':
            inp = data[17:18, :, :]
        elif self.targetMetric=='MK':
            inp = data[18:19, :, :]
        elif self.targetMetric == 'RK':
            inp = data[19:20, :, :]
        else:
            # Default is KFA
            inp = data[16:17, :, :]

        inp[np.isnan(inp)]=0
        # inp = torch.from_numpy(np.swapaxes(inp,0, 2))
        inp = torch.from_numpy(inp)

        # mu = torch.mean(inp, dim=(0, 1, 2), keepdim=True)
        # sd = torch.std(inp, dim=(0, 1, 2), keepdim=True)
        # img = (inp - mu) / sd

        # Scale data
        if self.targetMetric=='AK':
            inp[inp < 0] = 0 # Remove negative AK values
            inp[inp > 3.0] = 3.0 # Threshold values greater than 3.0
            inp = inp/3.0 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        elif self.targetMetric=='MK':
            inp[inp < 0] = 0 # Remove negative MK values
            inp[inp > 3.0] = 3.0 # Threshold values greater than 3.0
            inp = inp/3.0 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        elif self.targetMetric == 'RK':
            inp[inp < 0] = 0 # Remove negative RD values
            inp[inp > 3.0] = 3.0 # Threshold values greater than 3.0
            inp = inp/3.0 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        else:
            # Default is KFA, which is already between 0 and 1
            img = (inp - 0.5) / 0.5

        # Prep cond img
        # These are the conditioning DWIs
        b0 =  data[0:1,:,:]
        b1000 = data[1:self.numInputDWIs+1,:,:]
        b2000 = data[6:self.numInputDWIs+6,:,:]
        b3000 = data[11:self.numInputDWIs+11,:,:]
        inp = np.concatenate((b0,b1000,b2000,b3000), axis=0)
        inp = torch.from_numpy(inp)

        # Scale images
        cond_img = (inp - 0.5)/0.5

        ret['gt_image'] = img
        ret['cond_image'] = cond_img
        filename = os.path.split(path)
        # ret['path'] = re.sub('.npy', '.png', filename[1])
        ret['path'] = filename[1]
        return ret

    def __len__(self):
        return len(self.flist)

class DiffusionMRIDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, targetMetric='FA', numInputDWIs=6, image_size=[192, 192], loader=numpy_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                # transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = numpy_loader
        self.image_size = image_size
        self.targetMetric = targetMetric
        self.numInputDWIs = numInputDWIs

    def __getitem__(self, index):
        ret = {}
        path=self.data_root+self.flist[index]

        data = self.loader(path)

        # Prep input img
        if self.targetMetric=='ADC':
            inp = data[14:15, :, :]
        elif self.targetMetric=='AD':
            inp = data[15:16, :, :]
        elif self.targetMetric == 'RD':
            inp = data[16:17, :, :]
        else:
            # Default is FA
            inp = data[13:14, :, :]

        inp[np.isnan(inp)]=0
        # inp = torch.from_numpy(np.swapaxes(inp,0, 2))
        inp = torch.from_numpy(inp)

        # mu = torch.mean(inp, dim=(0, 1, 2), keepdim=True)
        # sd = torch.std(inp, dim=(0, 1, 2), keepdim=True)
        # img = (inp - mu) / sd

        # Scale data
        if self.targetMetric=='ADC':
            inp[inp < 0] = 0 # Remove negative ADC values
            inp[inp > 0.003] = 0.003 # Threshold values greater than 0.003
            inp = inp/0.003 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        elif self.targetMetric=='AD':
            inp[inp < 0] = 0 # Remove negative AD values
            inp[inp > 0.003] = 0.003 # Threshold values greater than 0.003
            inp = inp/0.003 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        elif self.targetMetric == 'RD':
            inp[inp < 0] = 0 # Remove negative RD values
            inp[inp > 0.003] = 0.003 # Threshold values greater than 0.003
            inp = inp/0.003 # Normalize to [0,1] range
            img = (inp - 0.5) / 0.5
        else:
            # Default is FA, which is already between 0 and 1
            img = (inp - 0.5) / 0.5


        # Prep cond img
        # These are the conditioning DWIs
        inp =  data[0:self.numInputDWIs+1,:,:]
        inp = torch.from_numpy(inp)

        # Scale images
        cond_img = (inp - 0.5)/0.5

        ret['gt_image'] = img
        ret['cond_image'] = cond_img
        filename = os.path.split(path)
        # ret['path'] = re.sub('.npy', '.png', filename[1])
        ret['path'] = filename[1]
        return ret

    def __len__(self):
        return len(self.flist)

#End Edit





