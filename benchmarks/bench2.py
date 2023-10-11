import torch
import torchvision
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset


def fix_seed(seed=0xBADCAFE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_seed()


model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()
transforms = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datasource, transforms: callable, length: int = 128):
        super().__init__()
        self.transforms = transforms
        self.length = length
        self.datasource = datasource

    def __len__(self) -> int:
        return len(self.datasource)

    def __getitem__(self, index: int) -> torch.Tensor:
        data = self.datasource[index]
        image, label = data['image'], data['label']
        if image.mode != 'RGB':
            image = Image.fromarray(np.array(image)[..., None].repeat(3, -1))
        return self.transforms(image), label


imagenette_valid = load_dataset('frgfm/imagenette', '320px', split='validation')

# %% [markdown]
# tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
# tiny_imagenet_valid = load_dataset('Maysee/tiny-imagenet', split='valid')


num_workers = 1
batch_size = 1


# trainset = Dataset(datasource=tiny_imagenet_train, transforms=transforms())
validset = Dataset(datasource=imagenette_valid, transforms=transforms())
valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
# valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, batch_size=batch_size)

def nmbytes(model: torch.nn.Module):
    n = 0
    for p in model.parameters():
        n += p.nbytes

    return n / 1024 ** 2


from torch.profiler import profile, record_function, ProfilerActivity




from itertools import product
from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import gc
from contextlib import nullcontext
from timeit import timeit
import time
from sklearn.metrics import accuracy_score


matmul_precision = ['medium', 'high', 'highest']
quantization = ['None', 'x86', 'fbgemm']
# mixed_precision = ['']
batch_sizes = [1]
num_workers = 1


def run_epoch(model, valid_dataloader, limit=2**32):
    Y = []
    Y_hat = []
    for i, (x, y) in enumerate(valid_dataloader):
        if i >= limit:
            break
        Y.append(y.item())
        y_hat = model(x).argmax(-1).item() > 0.0
        Y_hat.append(y_hat)
    return accuracy_score(np.array(Y), np.array(Y_hat))



limit = 256
T = {}
accuracy = {}
with torch.no_grad():
    for prec, quant, bs in tqdm(product(matmul_precision, quantization, batch_sizes)):
        valid_dataloader = torch.utils.data.DataLoader(validset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).eval()
        torch.set_float32_matmul_precision(prec)
        if quant != 'None':
            torch.backends.quantized.engine = quant
            qconfig_mapping = get_default_qconfig_mapping(quant)
            x = next(iter(valid_dataloader))[0]
            prepared_model = prepare_fx(model, qconfig_mapping, example_inputs=x)
            model = convert_fx(prepared_model)
        # if mp == '':
        #     amp = nullcontext()
        #     cast = lambda x: x
        # else:
        #     amp = torch.autocast(device_type='cpu', dtype=torch.bfloat16)
        #     cast = lambda x: x.to(torch.bfloat16)
        key  = '_'.join(map(str, [prec, quant, bs, round(nmbytes(model))]))
        # with amp:
        start = time.time()
        acc = run_epoch(model, valid_dataloader, limit)
        end = time.time()
            # t = timeit(lambda: run_epoch(model, valid_dataloader, limit, cast, globals), number=1) / limit
        T[key] = (end - start) / (limit / bs)
        accuracy[key] = acc
        gc.collect()


for key, t in T.items():
    print(key, t, accuracy[key])


