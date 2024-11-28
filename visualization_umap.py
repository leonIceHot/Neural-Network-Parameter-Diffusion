from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.models import resnet18
from models.resnet import ResNet18
from medmnist import INFO
import torch
import torch.nn as nn
import medmnist
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import umap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1.Preprocessing
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])

# 2.Load the dataset
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
        
        values     = DataClass(root=r'./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)                
        dataloader = DataLoader(values, batch_size=64, shuffle=False)       
                
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
                
        # Extract the feature
        features = []
        labels   = []
        with torch.no_grad():
            for images, target in dataloader:
                output = model(images).cpu()
                output = output.view(output.size(0), -1)  # 展平特征
                features.append(output.cpu().numpy())
                labels.append(target.cpu().numpy())
        
        features  = np.concatenate(features)
        labels    = np.concatenate(labels)
        
        #print("features shape",features.shape)
        #print("labels shape",labels.shape)
        
        # Visualize using the umap
        n    = 5
        dist = 0.05
        reducer   = umap.UMAP(n_neighbors=n, min_dist=dist, n_components=2, random_state=42, metric='euclidean')
        embedding  = reducer.fit_transform(features)
        #print(embedding.shape)
        
        # Save the image        
        plt.figure(dpi=300)
        plt.scatter(x=embedding[:,0],y=embedding[:,1],c=labels,cmap='Spectral',s=0.1)
        plt.colorbar(boundaries=np.arange(n_classes+1) - 0.5).set_ticks(np.arange(n_classes))        
        plt.savefig(f"/home/cluster/Pictures/New2/Visualization/Umap/{typename}/{dataset}_{typename}_UMAP.png")
        print(f"{dataset}_{typename}_UMAP")
        
        '''
        sns.scatterplot(x=embedding[:,:1].squeeze(axis=-1), 
                        y=embedding[:,1:].squeeze(axis=-1), 
                        s=10, alpha=0.8)
        '''        
        
        # matplotlib
        #plt.savefig(f"/home/cluster/Pictures/New2/Visualization/Umap/NND/{dataset}_NND_TSNE.png")
        #plt.savefig(f"/home/cluster/Pictures/New2/Visualization/Umap/Fine-Tuned/{dataset}_Fine-Tuned_UMAP.png")
        #plt.savefig(f"/home/cluster/Pictures/New2/Visualization/Umap/Co-Training/{dataset}_CO-Training_UMAP.png")
        plt.show()
        
        # seaborn
        #plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='Spectral', s=5)
        #plt.colorbar(boundaries=np.arange(n_classes+1) - 0.5).set_ticks(np.arange(n_classes))
        #plt.title(f'UMAP using {dataset}', fontsize=25, fontweight='bold')
