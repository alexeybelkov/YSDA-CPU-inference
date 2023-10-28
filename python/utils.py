import torch
import torchvision
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datasource, transforms: callable, labels: dict):
        super().__init__()
        self.transforms = transforms
        self.datasource = datasource
        self.labels = labels

    def __len__(self) -> int:
        return len(self.datasource)

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.datasource[index]
        image, label = data['image'], data['label']
        if image.mode != 'RGB':
            image = Image.fromarray(np.array(image)[..., None].repeat(3, -1))
        return self.transforms(image), self.labels[label]


def prepare_labels(path_to_labels: str = '../imagenet1000.txt'):
    with open(path_to_labels, 'r') as fopen:
        lines = fopen.readlines()

    def process_classes(line: str):
        splitted = line.strip().removeprefix('{').removesuffix(',').split(':')
        return (int(splitted[0]), splitted[1].strip().strip('\''))
    
    imagenette_classes = dict(enumerate(['tench', 'English springer', 'cassette player', 'chain saw', 
                                     'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']))

    orig_classes = dict(map(process_classes, lines))

    for k, v in imagenette_classes.items():
        for k1, v1 in orig_classes.items():
            if v in v1:
                imagenette_classes[k] = k1

    return imagenette_classes


def get_dataset(transforms: callable, path_to_labels: str = '../imagenet1000.txt'):
    imagenette_labels = prepare_labels(path_to_labels)
    imagenette_train = load_dataset('frgfm/imagenette', '320px', split='train')
    imagenette_valid = load_dataset('frgfm/imagenette', '320px', split='validation')
    trainset = Dataset(datasource=imagenette_train, transforms=transforms)
    validset = Dataset(datasource=imagenette_valid, transforms=transforms)
    return trainset, validset
