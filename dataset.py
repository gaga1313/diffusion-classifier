from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype
from PIL import Image
import os
import torch
import random

random.seed(43)

class ImageNet16(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        
        data = []
        with open('humancategory_info.json', 'r') as file:
            hc_info = json.load(file)
        file_paths = os.listdir(self.data_folder)
        unique_filter_levels = {}
        for fname in file_paths:
            f_splits = fname.split('_')
            filter = f_splits[3]
            if filter in unique_filter_levels:
               unique_filter_levels[filter].append(fname)
            else:
               unique_filter_levels[filter] = [fname]
        print(f'Unique filters : {unique_filter_levels.keys()}')
        nsample_per_level = 100//len(unique_filter_levels)
        file_paths = []
        for key,value in unique_filter_levels.items():
            random.shuffle(value)
            file_paths.extend(value[:nsample_per_level])
        random.shuffle(file_paths)
        print(file_paths)
        for fname in file_paths:
            f_splits = fname.split('_')
            category = f_splits[4]
            label = int(hc_info[category]['Hn_category'])
            file_path = os.path.join(self.data_folder, fname)
            data.append((str(file_path), label))
        return data
    
    def get_class_weights_and_distribution(self):

        data_distribution = {}
        class_weights = {}
        total_samples = len(self.data)
        class_weights = [0 for i in range(16)]
        for ele in self.data:
            _, label = ele
            if label in data_distribution:
                data_distribution[label]+=1
            else:data_distribution[label] = 1
        num_classes = len(data_distribution.keys())
        ideal_samples_class = int(total_samples/num_classes)  
        for key in data_distribution:       
            class_weights[int(key)] = round(ideal_samples_class/data_distribution[key], 3)
            data_distribution[key] = round(data_distribution[key] / total_samples, 3)      
        return class_weights, data_distribution
    
    def __numclasses__(self):
        classes = []
        for ele in self.data:
            _, label = ele
            if label not in classes:
                classes.append(label) 
        return classes, len(classes)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image_name = image_path.split('/')[-1]
        image = read_image(image_path, ImageReadMode.GRAY)
        image = convert_image_dtype(image, torch.float32)
        image = image.repeat(3,1,1)
        if self.transform:
            image = self.transform(image)
        return image_name, image, label
