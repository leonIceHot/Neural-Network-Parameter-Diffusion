from medmnist import BreastMNIST, ChestMNIST, PathMNIST, PneumoniaMNIST
from medmnist import INFO, Evaluator
import medmnist

#data_flag = 'dermamnist'
#data_flag = 'retinamnist'
#data_flag  = 'bloodmnist'

#data_flag  = 'tissuemnist'
#data_flag = 'breastmnist'
#data_flag = 'octmnist'

#data_flag = 'chestmnist'
#data_flag = 'pathmnist'
#data_flag = 'pneumoniamnist'

#data_flag  = 'organcmnist'
data_flag  = 'organsmnist'
#data_flag  = 'organamnist'

print(data_flag)
NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']

# Number of image channels
n_channels = info['n_channels']
print(f"channels:{n_channels}")

# Number of classes
n_classes = len(info['label'])
print(f"classes:{n_classes}")

# class name
class_names = info['label']
print(f"class name:{class_names}")


classlabel = []
for num, name in enumerate(class_names):
    #print(f'{num}')
    print(class_names[f'{num}'])
    classlabel.append(class_names[f'{num}'])

