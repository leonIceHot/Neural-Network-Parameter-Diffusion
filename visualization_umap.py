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
import matplotlib.font_manager as fm
import numpy as np
import umap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1.Preprocessing
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])


# 2.Load the dataset
#data1 = ['bloodmnist','dermamnist' ,'retinamnist','pneumoniamnist','octmnist','tissuemnist']
#data2 = ['BloodMNIST','DermaMNIST','RetinaMNIST','PneumoniaMNIST','OCTMNIST','TissueMNIST']

#data1 = ['breastmnist','chestmnist','pathmnist','organamnist', 'organcmnist','organsmnist']
#data2 = ['BreastMNIST','ChestMNIST','PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST']

#data1 = ['pathmnist','chestmnist']
#data2 = ['PathMNIST','ChestMNIST']

#types1 = ['NNPD','Fine-Tuned','CoTrain']
#types2 = ['Neural Network Diffusion','Transfer Learning','Temporal Difference Learning']

#lr = ['lr=0.1','lr=0.001', 'lr=0.0001']

types1  = ['TDL']
data    = ['organamnist','organcmnist','organsmnist']
data1   = ['OrganAMNIST','OrganCMNIST','OrganSMNIST']
lrrate  = ['lr1','lr2']

for num2, typename in enumerate(types1):
    for num, dataset in enumerate(data1):
        for num3, lr in enumerate(lrrate):
            info       = INFO[data[num]]
            task       = info['task']
            n_channels = info['n_channels']
            n_classes  = len(info['label'])
            DataClass  = getattr(medmnist, info['python_class'])

            values     = DataClass(root=r'./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)
            dataloader = DataLoader(values, batch_size=64, shuffle=False)

            # class name
            #class_names = info['label']

            # BloodMNIST
            if dataset == 'BloodMNIST':
                class_names = ['BAS','EOS','EBL','IMG','LYM','MON','NEU','PLT']

            # DermaMNIST
            if dataset == 'DermaMNIST':
                class_names = ['AKIEC','BCC','BKL','DF','MEL','NV','VASC']

            # RetinaMNIST
            if dataset == 'RetinaMNIST':
                class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

            # pneumoniamnist
            if dataset == 'pneumoniamnist':
                class_names = ['NORM','PNEU']

            # OCTMNIST
            if dataset == 'OCTMNIST':
                class_names = ['CNV','DME','DRU','NORM']

            # TissueMNIST
            if dataset == 'TissueMNIST':
                class_names = ['CDCT','DCT','GEC','IEC','LEU','POD','PTS','TAL']

            # BreastMNIST
            if dataset == 'BreastMNIST':
                class_names = ['MAL','NORM']

            # ChestMNIST
            if dataset == 'ChestMNIST':
                class_names = ['ATL', 'CDM', 'EFF', 'INF', 'MAS', 'NOD', 'PNEU', 'PTX', 'CON', 'EDM', 'EMP', 'FIB', 'PLR', 'HER']

            # PathMNIST
            if dataset == 'PathMNIST':
                class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'SM', 'NCM', 'CAS', 'CAE']

            # OrganAMNIST
            if dataset == 'OrganAMNIST':
                class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

            # OrganCMNIST
            if dataset == 'OrganCMNIST':
                class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

            # OrganSMNIST
            if dataset == 'OrganSMNIST':
                class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

            # print(f"class name:{class_names}")

            if typename == 'NNPD':
                path  = f"./param_data/MedMNIST/lr=0.001/{data1[num]}/data.pt"
                model = ResNet18(num_classes=n_classes, input_channel=n_channels)
                model.load_state_dict(torch.load(path),strict=False)
            elif typename == 'TL':
                path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/TL2/lr=0.001/" + data[num]
                model = torch.load(path)
            elif typename == 'TDL':
                path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/TDL/lr=0.001/{lr}/" + data[num]
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

            # 5. Label rename
            '''
            classlabel = []
            for num, name in enumerate(class_names):
                # print(f'{num}')
                print(class_names[f'{num}'])
                classlabel.append(class_names[f'{num}'])
            '''

            # Visualize using the umap
            n = 5
            dist = 0.05
            reducer = umap.UMAP(n_neighbors=n, min_dist=dist, n_components=2, metric='euclidean')
            embedding = reducer.fit_transform(features)

            # For only chestmnist
            if dataset == 'ChestMNIST':
                labels = np.random.randint(0, 14, 157066)  # 原始標籤數量過多，故需要修正
                labels = labels[:embedding.shape[0]]

            # Save the image
            plt.figure(dpi=300)
            plt.scatter(x=embedding[:,0],y=embedding[:,1],c=labels,cmap='Spectral',s=0.1)
            cbar = plt.colorbar(boundaries=np.arange(n_classes+1) - 0.5)
            cbar.set_ticks(np.arange(n_classes))

            italic_font = fm.FontProperties(style='italic')  # 設置字體為斜體
            cbar.set_ticklabels(class_names,fontproperties=italic_font,fontsize=10)
            print(f"{dataset}_{typename}_UMAP")
            plt.savefig(f"/home/cluster/Pictures/New6/Visualization/Umap/{typename}/{dataset}_{typename}_{lr}_UMAP.png")
            plt.show()


