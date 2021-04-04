import os
import random

from PIL import Image
from torch.utils import data
from torchvision import transforms

from datasets.transforms import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class IFFIDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.is_train = self.root.split("/")[-1] == "train"

    def __getitem__(self, index):
        if self.is_train:
            img_folder = os.path.join(self.root, sorted(os.listdir(self.root))[index])
            img_fname = sorted(os.listdir(img_folder))[10]
            filter_idx = random.randint(0, 16)
            filtered_img_fname = sorted(os.listdir(img_folder))[filter_idx]

            img = Image.open(os.path.join(img_folder, img_fname)).convert("RGB")
            filtered_img = Image.open(os.path.join(img_folder, filtered_img_fname)).convert("RGB")

            if self.transform:
                img, filtered_img = self.transform(img, filtered_img)

            return filtered_img, img, filter_idx
        else:
            img_folder = os.path.join(self.root, sorted(os.listdir(self.root))[index])
            img_fname = sorted(os.listdir(img_folder))[10]
            img = Image.open(os.path.join(img_folder, img_fname)).convert("RGB")

            filtered_img_fnames = sorted(os.listdir(img_folder))
            filtered_imgs = [Image.open(os.path.join(img_folder, filtered_img_fname)).convert("RGB") for filtered_img_fname in filtered_img_fnames]

            if self.transform:
                t_filtered_imgs, t_imgs = list(), list()
                for f_img in filtered_imgs:
                    im, f_img = self.transform(img, f_img)
                    t_filtered_imgs.append(f_img)
                    t_imgs.append(im)
                filtered_imgs, img = t_filtered_imgs, t_imgs

            return filtered_imgs, img

    def __len__(self):
        return len(os.listdir(self.root))


if __name__ == '__main__':
    iffi = IFFIDataset(root="../../../Downloads/IFFI-dataset/train",
                       transform=Compose([
                           # RandomCropTwoInstances(800, pad_if_needed=True),
                           # ResizeTwoInstances(512),
                           RandomHorizontalFlipTwoInstances(),
                           ToTensor()
                       ]))
    for i in range(len(iffi)):
        x, y, z, zn = iffi.__getitem__(i)
        try:
            if min(*x.size()[1:]) < 1081:
                print(i, x.size(), y.size(), z, zn)
        except:
            continue

    # transforms.ToPILImage()(x).show()
    # transforms.ToPILImage()(y).show()
