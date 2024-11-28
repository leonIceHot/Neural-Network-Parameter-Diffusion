from MulticoreTSNE import MulticoreTSNE as TSNE
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from torchvision.models import resnet18
from models.resnet import ResNet18
from medmnist import INFO, Evaluator
import MulticoreTSNE
import medmnist
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. Preprocessing
# EMNIST
'''
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,),(0.30810776,))])

emnist_dataset = datasets.EMNIST(root='./data2', split='balanced', train=False, download=True, transform=transform)
dataloader = DataLoader(emnist_dataset, batch_size=64, shuffle=False)
'''

# MNIST
'''
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])


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
# 2. Load the dataset
data1  = ['breastmnist','octmnist','organamnist','organcmnist','organsmnist','pneumoniamnist','tissuemnist']
data2  = ['BreastMNIST','OCTMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST','PneumoniaMNIST','TissueMNIST']

#types1 = ['NND','Fine-Tuned','CoTrain']
#types2 = ['Neural Network Diffusion','Transfer Learning','Temporal Difference Learning']

types1 = ['Fine-Tuned','CoTrain']
types2 = ['Transfer Learning','Temporal Difference Learning']


for num2, typename in enumerate(types1):
    for num, dataset in enumerate(data2):
        info       = INFO[data1[num]]
        task       = info['task']        
        n_channels = info['n_channels']
        n_classes  = len(info['label'])            
        DataClass  = getattr(medmnist, info['python_class'])
        
        values     = DataClass(root='./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)                
        dataloader = DataLoader(values, batch_size=64, shuffle=False)
        
        # 3. Load the model
        if typename == 'NND':        
            path  = "./param_data/MedMNIST/"+ dataset +"/data.pt"
            model = ResNet18(num_classes=n_classes, input_channel=n_channels)
            model.load_state_dict(torch.load(path),strict=False)
        elif typename == 'Fine-Tuned':
            path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/Fine-Tuned/" + data1[num]
            model = torch.load(path)
        elif typename == 'CoTrain':
            path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/CoTrain/" + data1[num]
            model = torch.load(path)
                            
        model.cpu()
        model.eval()
        
        # 3. Extract the feature
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
        #tsne = TSNE(n_jobs=24)
        fit_tsne = MulticoreTSNE.MulticoreTSNE(n_jobs=1, random_state=42)
        features_2d = fit_tsne.fit_transform(features)
         
        # 5. 可视化 t-SNE 结果
        plt.figure(dpi=300)
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='Spectral', s=0.1)
        plt.colorbar(boundaries=np.arange(n_classes+1) - 0.5).set_ticks(np.arange(n_classes))
        '''
        sns.scatterplot(x=features_2d[:,:1].squeeze(axis=-1),
                        y=features_2d[:,1:].squeeze(axis=-1),
                        s=10, alpha=0.8)
        '''
        print(f"{dataset}_{typename}_TSNE")
        plt.savefig(f"/home/cluster/Pictures/New2/Visualization/TSNE/{typename}/{dataset}_{typename}_t-SNE.png")                
        plt.show()
    
