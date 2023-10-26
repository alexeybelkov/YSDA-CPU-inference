import torch
import torchvision
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing



torch.multiprocessing.set_sharing_strategy('file_system')



model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()
transforms = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms


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
    def __init__(self, datasource, transforms: callable, ramming: bool = False):
        super().__init__()
        self.transforms = transforms
        self.ramming = ramming
        if ramming:
            ram_data = []
            for i in range(len(datasource)):
                data = datasource[i]
                ram_data.append({'image': data['image'], 'label': data['label']})
            self.datasource = ram_data
        else:
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



num_workers = 1
batch_size = 1

t = Dataset(datasource=tiny_imagenet_train, transforms=transforms())
tf = transforms()
trainset = Dataset(datasource=imagenette_train, transforms=tf)
validset = Dataset(datasource=imagenette_valid, transforms=tf, ramming=True)
valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
# valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, batch_size=batch_size)


def nbytes(model: torch.nn.Module):
    n = 0
    for p in model.parameters():
        n += p.nbytes

    return n / 1024 ** 2


nbytes(model)


from torch.profiler import profile, record_function, ProfilerActivity
from itertools import product
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import gc
from contextlib import nullcontext
from timeit import timeit
import time
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import datetime

def fix_seed(worker_id=0, seed=0xBADCAFE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed()

torch_generator = torch.Generator()
torch_generator.manual_seed(0xBADCAFFE)


examples_inputs = [validset[i][0] for i in range(len(validset) // 100)]

quantization = ['None', 'x86', 'fbgemm']
mixed_precision = ['']
batch_sizes = [1, 4]
num_workers = 1


def run_epoch(model, valid_dataloader, limit=2**32):
    T = 0.0
    Y = []
    Y_hat = []
    for i, (x, y) in enumerate(valid_dataloader):
        if i >= limit:
            break
        Y.append(y.ravel())
        start = time.time()
        y_hat = model(x)
        end = time.time()
        Y_hat.append(y_hat.argmax(-1))
        T += end - start
    return accuracy_score(np.array(Y).ravel(), np.array(Y_hat).ravel()), T

def calibrate(model, dataloader):
    with torch.no_grad():
        for x, y in dataloader:
            model(x)

limit = 16
T = {}
date_time = datetime.datetime.now()
accuracy = {}
with torch.no_grad():
    with open(f'profiling{date_time}.txt', 'w+') as fopen:
        for quant, bs in tqdm(product(quantization, batch_sizes)):
            valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, 
                                                        batch_size=batch_size, shuffle=True, 
                                                        worker_init_fn=fix_seed, generator=torch_generator)
            model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).eval()
            if quant != 'None':
                torch.backends.quantized.engine = quant
                qconfig_mapping = get_default_qconfig_mapping(quant)
                prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=next(iter(valid_dataloader))[0])
                calibrate(prepared_model, valid_dataloader)
                model = convert_fx(prepared_model)
            key  = '_'.join(map(str, [quant, bs, round(nbytes(model))]))
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                for i, (x, y) in enumerate(valid_dataloader):
                    with record_function("model_inference"):
                        if i >= limit:
                            break
                        model(x)
            fopen.write(f'{key}\n')
            fopen.write(prof.key_averages().table(sort_by="cpu_time_total"))
            gc.collect()
