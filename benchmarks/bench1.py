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
grid_batch_size = [1, 2, 4, 6]



results = {}

with torch.no_grad():
    for num_workers, batch_size in tqdm(product(grid_num_workers, grid_batch_size)):
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)
        results[(num_workers, batch_size)] = 0
        for x, y in dataloader:
            t = timeit.timeit(lambda: model(x), number=1)
            results[(num_workers, batch_size)] += t
        results[(num_workers, batch_size)] /= (len(dataset) / batch_size)
        # print(t)

df = pd.DataFrame(index=grid_num_workers, columns=grid_batch_size)

for (n, b), t in results.items():
    df[n, b] = t
    
print(df)

df.to_csv('default-vitb16.csv', index=df.index)