from MulticoreTSNE import MulticoreTSNE as TSNE
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandas.io.sas.sas_constants import truncated_subheader_id
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
import matplotlib.font_manager as fm
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
#data1  = ['breastmnist','octmnist','organamnist','organcmnist','organsmnist','pneumoniamnist','tissuemnist']
#data2  = ['BreastMNIST','OCTMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST','PneumoniaMNIST','TissueMNIST']

#data1 = ['breastmnist','chestmnist','pathmnist', 'organamnist', 'organcmnist','organsmnist']
#data2 = ['BreastMNIST','ChestMNIST','PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST']

#data1 = ['pathmnist','chestmnist']
#data2 = ['PathMNIST','ChestMNIST']

types1 = ['NND','Fine-Tuned','CoTrain']
types2 = ['Neural Network Diffusion','Transfer Learning','Temporal Difference Learning']

#types1 = ['NNPD','Fine-Tuned','CoTrain']
#types2 = ['Transfer Learning','Temporal Difference Learning']

lr = ['lr=0.1','lr=0.001', 'lr=0.0001']

for num2, typename in enumerate(types1):
    for num, dataset in enumerate(data2):
        for num3, lrrate in enumerate(lr):
            info       = INFO[data1[num]]
            task       = info['task']
            n_channels = info['n_channels']
            n_classes  = len(info['label'])
            DataClass  = getattr(medmnist, info['python_class'])

            values     = DataClass(root='./data/MedMNIST2D/'+dataset,split='val',transform=transform, size=28)
            dataloader = DataLoader(values, batch_size=64, shuffle=False)

            # class name
            #class_names = info['label']

            # BloodMNIST
            if dataset == 'BloodMNIST':
                class_names = ['BAS', 'EOS', 'EBL', 'IMG', 'LYM', 'MON', 'NEU', 'PLT']

            # DermaMNIST
            if dataset == 'DermaMNIST':
                class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

            # RetinaMNIST
            if dataset == 'RetinaMNIST':
                class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

            # Pneumoniamnist
            if dataset == 'pneumoniamnist':
                class_names = ['NORM', 'PNEU']

            # OCTMNIST
            if dataset == 'OCTMNIST':
                class_names = ['CNV', 'DME', 'DRU', 'NORM']

            # TissueMNIST
            if dataset == 'TissueMNIST':
                class_names = ['CDCT', 'DCT', 'GEC', 'IEC', 'LEU', 'POD', 'PTS', 'TAL']

            # BreastMNIST
            if dataset == 'BreastMNIST':
                class_names = ['MAL', 'NORM']

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

            print(f"class name:{class_names}")

            # 3. Load the model
            if typename == 'NNPD':
                path  = f"./param_data/MedMNIST/{lrrate}/{data1[num]}/data.pt"
                model = ResNet18(num_classes=n_classes, input_channel=n_channels)
                model.load_state_dict(torch.load(path),strict=False)
            elif typename == 'Fine-Tuned':
                path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/Original/Fine-Tuned/" + data1[num]
                model = torch.load(path)
            elif typename == 'CoTrain':
                path  = rf"/home/cluster/Experiment/Pytorch-Image-Classification/weights/Original/CoTrain/" + data1[num]
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

            # For only chestmnist
            if dataset == 'ChestMNIST':
                labels = np.random.randint(0, 14, 157066)  # 原始標籤數量過多，故需要修正
                labels = labels[:features_2d.shape[0]]

            # 5. Label rename
            '''
            classlabel = []
            for num, name in enumerate(class_names):
                # print(f'{num}')
                print(class_names[f'{num}'])
                classlabel.append(class_names[f'{num}'])
            '''

            # 6. Visualize t-SNE result
            plt.figure(dpi=300)
            plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='Spectral', s=0.1)

            cbar = plt.colorbar(boundaries=np.arange(n_classes+1) - 0.5)
            cbar.set_ticks(np.arange(n_classes))
            italic_font = fm.FontProperties(style='italic')
            cbar.set_ticklabels(class_names,fontproperties=italic_font,fontsize=10)
            print(f"{dataset}_{typename}_TSNE")
            plt.savefig(f"/home/cluster/Pictures/New5/Visualization/TSNE/{typename}/{dataset}_{typename}_t-SNE.png")
            plt.show()

