import torch
import os

# https://github.com/pytorch/pytorch/issues/19001

print('DEFAULT:', torch.get_num_threads(), torch.get_num_interop_threads())
torch.set_num_threads(1), torch.set_num_interop_threads(1)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
print(torch.get_num_threads(), torch.get_num_interop_threads())
print(torch.__config__.parallel_info())

import torchvision
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing
from torch import nn
from matplotlib import pyplot as plt


torch.multiprocessing.set_sharing_strategy('file_system')

model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
model.eval()
transforms = torchvision.models.ResNet34_Weights.IMAGENET1K_V1.transforms

with open('../imagenet1000.txt', 'r') as fopen:
    lines = fopen.readlines()

def process_classes(line: str):
    splitted = line.strip().removeprefix('{').removesuffix(',').split(':')
    return (int(splitted[0]), splitted[1].strip().strip('\''))

orig_classes = dict(map(process_classes, lines))

imagenette_classes = dict(enumerate(['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']))

for k, v in imagenette_classes.items():
    for k1, v1 in orig_classes.items():
        if v in v1:
            imagenette_classes[k] = k1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datasource, transforms: callable):
        super().__init__()
        self.transforms = transforms
        self.datasource = datasource

    def __len__(self) -> int:
        return len(self.datasource)

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.datasource[index]
        image, label = data['image'], data['label']
        if image.mode != 'RGB':
            image = Image.fromarray(np.array(image)[..., None].repeat(3, -1))
        return self.transforms(image), imagenette_classes[label]
    

imagenette_train = load_dataset('frgfm/imagenette', '320px', split='train')
imagenette_valid = load_dataset('frgfm/imagenette', '320px', split='validation')


batch_size = 1
tf = transforms()
trainset = Dataset(datasource=imagenette_train, transforms=tf)
validset = Dataset(datasource=imagenette_valid, transforms=tf)
valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=0, batch_size=batch_size, shuffle=False)

def nbytes(model: torch.nn.Module):
    n = 0
    for p in model.parameters():
        n += p.nbytes
    return n / 1024 ** 2

from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import DeQuantStub, QuantStub
from torch.ao.quantization import QConfigMapping
import gc
from contextlib import nullcontext
from timeit import timeit
import time
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import datetime
import torch.quantization._numeric_suite_fx as ns

def fix_seed(worker_id=0, seed=0xBADCAFE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed()

torch_generator = torch.Generator()
torch_generator.manual_seed(0xBADCAFFE)

from tqdm import tqdm
from copy import deepcopy
from typing import List, Optional


copy_model = deepcopy(model)
module_a = deepcopy(model)
module_a.fc = nn.Identity()
module_b = model.fc

qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)
example_inputs_a = next(iter(valid_dataloader))[0]
example_inputs_b = torch.randn(1, 512)
prepared_module_a = prepare_fx(module_a, qconfig_mapping, example_inputs_a)
prepared_module_b = prepare_fx(module_b, qconfig_mapping, example_inputs_b)

with torch.inference_mode():
    for x, _ in tqdm(valid_dataloader):
        y = prepared_module_a(x)
        prepared_module_b(y)


quantized_a = convert_fx(prepared_module_a) 
quantized_b = convert_fx(prepared_module_b) 

gt = []
pred = []
embeddings = []
quant_embeddings = []
Y = []
with torch.inference_mode():
    for x, y in tqdm(valid_dataloader):
        qemb = quantized_a(x)
        quant_embeddings.append(qemb)
        embeddings.append(module_a(x))
        y_hat = quantized_b(qemb)
        Y.append(y_hat)
        gt.append(y)
        pred.append(y_hat.argmax(-1))
    gt = torch.cat(gt).ravel().numpy()
    pred = torch.cat(pred).ravel().numpy()

gc.collect()

print('quantized accuracy:', accuracy_score(gt, pred))

batch_size = 1
epochs = 128

times = []

with torch.inference_mode():
    for _ in tqdm(range(10)):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for epoch in range(epochs):
                indexes = np.random.randint(0, len(validset), size=batch_size)
                batch = torch.cat([quant_embeddings[i] for i in indexes])
                with record_function("quantized_inference"):
                    quantized_b(batch)
                batch = torch.cat([embeddings[i] for i in indexes])
                with record_function("float_inference"):
                    module_b(batch)
        pair = [-999, -999]
        for item in list(prof.profiler.key_averages()):
            if item.key == 'quantized::linear':
                pair[0] = item.cpu_time_total
            elif item.key == 'aten::linear':
                pair[1] = item.cpu_time_total
        times.append(pair)

quantized_time = [t[0] for t in times]
float_time = [t[1] for t in times]
batch_sizes = [2 ** i for i in range(10)]

plt.figure(figsize=(15, 10))
plt.plot(batch_sizes, quantized_time, '--bo', label='quant')
plt.plot(batch_sizes, float_time, '--ro', label='aten')
plt.legend()
plt.title('quatized::linear vs aten::linear')
plt.xlabel('batch_size')
plt.ylabel('microseconds')
plt.savefig('../bench.jpg')
        