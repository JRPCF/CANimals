"""
Pytorch dataset class for the Stanford Dogs dataset
"""
"""
Yarne Hermann YPH2105
"""

import os
import random
random.seed(0)

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class StanfordDogs(Dataset):
    NUM_CLASSES = 120
    CROP_SIZE = 256

    def __init__(self, path, specific_classes=None):
        super().__init__()
        self.dataset_dict, self.classes = self.load_data_from_path(path)
        self.specific_classes = specific_classes
        self.current_dataset = self.prepare_dataset_for_use()
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.CROP_SIZE, self.CROP_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def load_data_from_path(self, path):
        # 1 Read class names from directories and store in a classes array
        # Taking [10:] makes us get the dog name without a code in the front
        # classes = [o[10:] for o in os.listdir(path) if os.path.isdir(os.path.join(path,o))]
        classes = []
        dataset_dict = {}

        # 2 Read in the images, with corresponding label the index of the class name
        for o in os.listdir(path):
            if os.path.isdir(os.path.join(path, o)):
                class_name = o[10:] # Taking [10:] makes us get the dog name without a code in the front
                classes.append(class_name)
                dataset_dict[class_name] = []
                for img_path in os.listdir(os.path.join(path, o)):
                    dataset_dict[class_name].append(os.path.join(path, o, img_path))
        return dataset_dict, classes

    # 3 construct current (shuffled) dataset
    def prepare_dataset_for_use(self):
        current_dataset = []
        if self.specific_classes is None:
            for class_name, class_image_paths in self.dataset_dict.items():
                class_idx = self.classes.index(class_name)
                for img_path in class_image_paths:
                    current_dataset.append((img_path, class_idx))
        else: #only get specific classes
            for class_name in self.specific_classes:
                class_image_paths = self.dataset_dict[class_name]
                class_idx = self.classes.index(class_name)
                for img_path in class_image_paths:
                    current_dataset.append((img_path, class_idx))

        # shuffle
        random.shuffle(current_dataset)
        return current_dataset



    """
    This will allow to restrict to only a subset of more specific classes
    """
    def set_specific_classes(self, specific_classes=None):
        # self.specific_classes=specific_classes
        # #TODO: adapt current_dataset to only use images from the specific classes
        # self.current_dataset = self.prepare_dataset_for_use()
        pass

    def get_num_classes(self):
        return self.NUM_CLASSES

    def __getitem__(self, index):
        img_path, label = self.current_dataset[index]
        image = Image.open(img_path)
        if (np.array(image).shape[2] != 3):
            print(img_path, label, np.array(image).shape)
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.current_dataset)

