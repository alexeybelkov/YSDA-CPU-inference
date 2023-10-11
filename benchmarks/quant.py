import torch
import torchvision
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
from itertools import product
import timeit
from functools import partial
import pandas as pd


def fix_seed(seed=0xBADCAFE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed()


model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
model.eval()
transforms = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1.transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transforms: callable, length: int = 128):
        super().__init__()
        self.transforms = transforms
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.fromarray(np.random.randint(0, 2**8, size=(256, 256, 3), dtype=np.uint8))
        return self.transforms(image), np.random.binomial(1, 0.5)


dataset = Dataset(transforms=transforms())

grid_num_workers = [1, 2, 4, 6, 8]
grid_batch_size = [1, 2, 4, 6, 8]



results = {}

from torch.ao.quantization import get_default_qconfig_mapping
from torch.quantization.quantize_fx import prepare_fx, convert_fx

qconfig_mapping = get_default_qconfig_mapping()
# Or explicity specify the qengine
# qengine = 'x86'
# torch.backends.quantized.engine = qengine
# qconfig_mapping = get_default_qconfig_mapping(qengine)

model_fp32 = model.eval()
x = torch.randn((1, 3, 224, 224), dtype=torch.float)
x = x.to(memory_format=torch.channels_last)

# Insert observers according to qconfig and backend config
prepared_model = prepare_fx(model_fp32, qconfig_mapping, example_inputs=x)

# Calibration code not shown

# Convert to quantized model
quantized_model = convert_fx(prepared_model)


with torch.no_grad():
    for num_workers, batch_size in tqdm(product(grid_num_workers, grid_batch_size)):
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
        results[(num_workers, batch_size)] = 0
        for x, y in dataloader:
            t = timeit.timeit(lambda: model(x), number=1)
            results[(num_workers, batch_size)] += t
        results[(num_workers, batch_size)] /= len(dataset)
        # print(t)

df = pd.DataFrame(index=grid_num_workers, columns=grid_batch_size)

for (n, b), t in results.items():
    df[n, b] = t
    
print(df)

df.to_csv('default-vitb16.csv', index=df.index)