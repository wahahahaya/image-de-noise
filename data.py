import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class img_data(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root,"%s/ground" % mode)+"/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root,"%s/gauss" % mode)+"/*.*"))
        self.files_C = sorted(glob.glob(os.path.join(root,"%s/sp" % mode)+"/*.*"))
        #self.files_D = sorted(glob.glob(os.path.join(root,"%s/ground" % mode)+"/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_B = Image.open(self.files_B[index % len(self.files_B)])
        image_C = Image.open(self.files_C[index % len(self.files_C)])


        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)
        return {"groundtruth": item_A, "gauss": item_B, "sp": item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))