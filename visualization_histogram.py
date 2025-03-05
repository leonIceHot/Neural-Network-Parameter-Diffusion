from medmnist import INFO
import matplotlib.pyplot as plt
import numpy as np
import medmnist
import random
import matplotlib as mpl
import seaborn as sns
import pandas as pd


# 全局字體設置
'''
mpl.rcParams['font.family']     = 'sans-serif'         # 設置字體
mpl.rcParams['font.size']       = 12                   # 全局字體大小
mpl.rcParams['axes.titlesize']  = 16                   # 標題字體大小
mpl.rcParams['axes.labelsize']  = 6                    # 軸標籤字體大小
mpl.rcParams['xtick.labelsize'] = 6                    # X 軸刻度字體大小
mpl.rcParams['ytick.labelsize'] = 8                    # Y 軸刻度字體大小
mpl.rcParams['legend.fontsize'] = 6                    # 圖例字體大小
'''

# Config
#label = 'multi-label'
label = 'multi-class'

if label == 'multi-class':
    dataset1 = ['dermamnist','retinamnist','bloodmnist','tissuemnist','octmnist','pathmnist','organamnist','organcmnist','organsmnist','breastmnist', 'pneumoniamnist']
    dataset2 = ['DermaMNIST','RetinaMNIST','BloodMNIST','TissueMNIST','OCTMNIST','PathMNIST','OrganAMNIST','OrganCMNIST','OrganSMNIST','BreastMNIST', 'PneumoniaMNIST']
    #dataset1 = ['breastmnist', 'pneumoniamnist']
    #dataset2 = ['BreastMNIST', 'PneumoniaMNIST']

if label == 'multi-label':
    dataset1 = ['chestmnist']
    dataset2 = ['ChestMNIST']


# 隨機生成顏色的函數
def generate_random_colors(num_colors):
    # 隨機生成十六進位的顏色碼
    return ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(num_colors)]

# 選擇特定的 medmnist 資料集，如 "pathmnist"
for num,data in enumerate(dataset1):
    DataClass = getattr(medmnist, INFO[dataset1[num]]['python_class'])

    # 隨機生成顏色的函數
    def generate_random_colors(num_colors):
        # 隨機生成十六進位的顏色碼
        return ['#%06x' % random.randint(0, 0xFFFFFF) for _ in range(num_colors)]
    print(f"dataset:{dataset1[num]}")
    print(f"dataset2:{dataset2[num]}")

    #dataset2 = dataset2[num]

    # 自動下載並載入 train、val、test 資料集
    print(f'./data/MedMNIST2D/{dataset2[num]}')
    train_dataset  = DataClass(root=f'./data/MedMNIST2D/{dataset2[num]}',split='train',download=True)
    val_dataset    = DataClass(root=f'./data/MedMNIST2D/{dataset2[num]}',split='val', download=True)
    test_dataset   = DataClass(root=f'./data/MedMNIST2D/{dataset2[num]}',split='test', download=True)

    # 將所有標籤集結起來
    train_labels   = train_dataset.labels
    val_labels     = val_dataset.labels
    test_labels    = test_dataset.labels

    all_labels     = np.concatenate([train_labels, val_labels, test_labels], axis=0)

    # all_labels 通常為 (N,1) 的矩陣，先壓平成一維陣列
    all_labels     = all_labels.squeeze()


    # BloodMNIST
    if dataset2[num] == 'BloodMNIST':
        class_names = ['BAS', 'EOS', 'EBL', 'IMG', 'LYM', 'MON', 'NEU', 'PLT']


    # DermaMNIST
    if dataset2[num] == 'DermaMNIST':
        class_names = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

    # RetinaMNIST
    if dataset2[num] == 'RetinaMNIST':
        class_names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'PDR']

    # PneumoniaMNIST
    if dataset2[num] == 'PneumoniaMNIST':
        class_names = ['NORM', 'PNEU']

    # OCTMNIST
    if dataset2[num] == 'OCTMNIST':
        class_names = ['CNV', 'DME', 'DRU', 'NORM']

    # TissueMNIST
    if dataset2[num] == 'TissueMNIST':
        class_names = ['CDCT', 'DCT', 'GEC', 'IEC', 'LEU', 'POD', 'PTS', 'TAL']

    # BreastMNIST
    if dataset2[num] == 'BreastMNIST':
        class_names = ['MAL', 'NORM']

    # ChestMNIST
    if dataset2[num] == 'ChestMNIST':
        class_names = ['ATL', 'CDM', 'EFF', 'INF', 'MAS', 'NOD', 'PNEU', 'PTX', 'CON', 'EDM', 'EMP', 'FIB', 'PLR', 'HER']

    # PathMNIST
    if dataset2[num] == 'PathMNIST':
        class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'SM', 'NCM', 'CAS', 'CAE']

    # OrganAMNIST
    if dataset2[num] == 'OrganAMNIST':
        class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

    # OrganCMNIST
    if dataset2[num] == 'OrganCMNIST':
        class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

    # OrganSMNIST
    if dataset2[num] == 'OrganSMNIST':
        class_names = ['BLD', 'FEM-L', 'FEM-R', 'HRT', 'KID-L', 'KID-R', 'LIV', 'LNG-L', 'LNG-R', 'PAN', 'SPL']

    # 設置 Seaborn 主題
    sns.set_theme(style="darkgrid", palette="pastel",context="paper",font="sans-serif")
    plt.tick_params(axis='both', which='both', length=1, width=1, bottom=True, left=True)

    # For multi-label
    if label == 'multi-label':
        num_classes    = all_labels.shape[1]
        class_ids      = np.arange(num_classes)
        counts         = np.sum(all_labels, axis=0)
        labels         = len(class_names)

        print(f"class name:{class_names}")
        print(f"counts:{counts}")
        ax = sns.barplot(x=class_names,y=counts,color='#4C72B0')

        for patch in ax.patches:
            patch.set_width(0.5)

        # Seaborn
        #colors = sns.color_palette("pastel", n_colors=labels)
        # plt.bar(labels, counts, color=colors)
        #sns.barplot(x=labels,y=counts,color=colors,palette='viridis')


        # Matplotlib
        #colors         = generate_random_colors(labels)
        #plt.bar(class_ids, counts, width=0.5, color=colors, edgecolor='black')

        x = range(labels)
        plt.xticks(ticks=x, labels=class_names, fontsize=8)
        #plt.tick_params(axis='both', which='both', length=5, width=1)

    # For multi-class
    if label == 'multi-class':
        unique, counts = np.unique(all_labels, return_counts=True)
        labels         = len(class_names)

        print(f"class name:{class_names}")
        print(f"counts:{counts}")
        ax = sns.barplot(x=class_names, y=counts,color='#4C72B0')

        # 想要將寬度調整為0.5
        desired_width = 0.5 # 0.3
        # 假設原始寬度0.8（Seaborn默認）
        original_width = 0.8

        # 逐一處理每個柱子 (patch)
        for patch in ax.patches:
            # 1) 調整寬度
            patch.set_width(desired_width)

            # 2) 為了維持柱子與原x軸刻度中心對齊，需要手動「右移」(或左移)一些距離
            #    差值的一半 = (原寬度 - 新寬度) / 2
            current_x = patch.get_x()
            shift = (original_width - desired_width) / 2
            patch.set_x(current_x + shift)

        # Seaborn
        #colors = sns.color_palette("pastel", n_colors=labels)
        #plt.bar(labels, counts, color=colors)
        #sns.barplot(x=labels, y=counts, color=colors)

        # Matplotlib
        #colors = generate_random_colors(labels)
        #plt.bar(unique, counts, width=0.5, color=colors, edgecolor='black')

        x = range(labels)
        plt.xticks(ticks=x, labels=class_names, fontsize=8)
        #plt.tick_params(axis='both', which='both', length=5, width=1)

    # 繪製類別分佈直方圖
    plt.grid(axis='both', which='both', color='white', linestyle='-', linewidth=0.5)
    plt.title(f'{dataset2[num]}', fontname='sans-serif')
    plt.savefig(f"/home/cluster/Pictures/New5/Visualization/Histogram/{dataset2[num]}.png",dpi=300)
    plt.show()
