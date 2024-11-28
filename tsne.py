from MulticoreTSNE import MulticoreTSNE as TSNE
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from models.resnet import ResNet18
from medmnist import INFO, Evaluator
import medmnist
import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 EMNIST 数据集
'''
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,),(0.30810776,))])

emnist_dataset = datasets.EMNIST(root='./data2', split='balanced', train=False, download=True, transform=transform)
dataloader = DataLoader(emnist_dataset, batch_size=64, shuffle=False)
'''

'''
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307), (0.3081))])
mnist_dataset = datasets.MNIST(root='./data2', train=False, download=True, transform=transform)
dataloader = DataLoader(mnist_dataset, batch_size=64, shuffle=False)
'''

# MedMNIST
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

# 2. 加载预训练的 ResNet 模型
# pth
#n_channels = 1
#n_classes  = 10
#dataset    = "mnist"
#path    = "./param_data/MedMNIST/"+ dataset +"/data.pt"
data_flag  = 'organsmnist'
info       = INFO[data_flag]
task       = info['task']        
n_channels = info['n_channels']
n_classes  = len(info['label'])            
DataClass  = getattr(medmnist, info['python_class'])

dataset    = 'OrganSMNIST'
values     = DataClass(root='./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)                
dataloader = DataLoader(values, batch_size=64, shuffle=False)
path       = "./param_data/MedMNIST/"+ dataset +"/data.pt"
model      = ResNet18(num_classes=n_classes, input_channel=n_channels)
model.load_state_dict(torch.load(path),strict=False)
model.eval()

# 3. 提取特征和标签
features = []
labels = []

with torch.no_grad():
    for images, target in dataloader:
        output = model(images)
        output = output.view(output.size(0), -1)  # 展平特征
        features.append(output.cpu().numpy())
        labels.append(target.cpu().numpy())

features = np.concatenate(features)
labels   = np.concatenate(labels)

# 4. 使用 t-SNE 进行降维
#tsne = TSNE(n_components=2, random_state=42)
#features_2d = tsne.fit_transform(features)
tsne = TSNE(n_jobs=24)
features_2d = tsne.fit_transform(features)

# 5. 可视化 t-SNE 结果
plt.figure(dpi=300)
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
plt.colorbar(scatter)
plt.title("t-SNE of "+ dataset + " using ResNet-18 features")
plt.savefig(f"/home/cluster/Pictures/nnd/TSNE/MedMNIST/{dataset}.png")
plt.show()

