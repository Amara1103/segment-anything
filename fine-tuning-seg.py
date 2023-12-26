from segment_anything import sam_model_registry, SamPredictor
import torch
from data_loading import BasicDataset, CarvanaDataset
import os
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from pathlib import Path
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

# 加载模型
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 设置超参数
img_scale=0.5
val_percent=0.1
batch_size=8
num_epochs=5


# 1. Create dataset
try:
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

# 2. Split into train / validation partitions
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. Create data loaders
loader_args = dict(batch_size=batch_size,
                    num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False,
                        drop_last=True, **loader_args)

# 修改损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(sam.parameters(), lr=0.001, momentum=0.9)


for epoch in range(num_epochs):
    for inputs, labels in train_loader:  # 使用你的数据加载器
        optimizer.zero_grad()
        outputs = sam(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
torch.save({
    'epoch': num_epochs,
    'model_state_dict': sam.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    }, f'{dir_checkpoint}segment_anything_nesting.pth')