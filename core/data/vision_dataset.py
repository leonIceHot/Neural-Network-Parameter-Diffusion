from .base import DataBase
from medmnist import BreastMNIST, ChestMNIST, PathMNIST
from medmnist import INFO, Evaluator
import medmnist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import torch
import timm

class VisionData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(VisionData, self).__init__(cfg, **kwargs)

        """
        init data for classification task
        we firstly load the dataset and then define the transform for the dataset
        args:
            cfg: the config file
        
        cfg args:
            data_root: the root path of the dataset
            dataset: the dataset name
            batch_size: the batch size
            num_workers: the number of workers
            
        """
        
        super(VisionData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.dataset = getattr(self.cfg, 'dataset', 'mnist')
        self.dataset2 = 'rfmid'

        # MedMNIST
        self.data_flag  = 'chestmnist'
        self.info       = INFO[self.data_flag]
        self.task       = self.info['task']        
        self.n_channels = self.info
        self.n_classes  = len(self.info['label'])            
        self.DataClass  = getattr(medmnist, self.info['python_class'])

    @property
    def train_transform(self):
        train_transform = {
            'cifar10': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))]),
            
            'cifar100': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))]),
            
            'mnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))]),
            
            'cinic10': transforms.Compose([
                    transforms.Resize(size=(28,28)),
                    transforms.RandomHorizontalFlip(),                                   
                    transforms.ToTensor(),
                    transforms.Normalize((0.47889522,0.47227842,0.43047404),
                                         (0.24205776,0.23828046,0.25874835))]),
            
            'oxfordIIITPets': transforms.Compose([
                    transforms.Resize(size=(28,28)),
                    transforms.RandomHorizontalFlip(),                                   
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),
                                         (0.229,0.224,0.225))]),            
            
            'medmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))]),
            
            'kmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))]),
            
            'emnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.13066062,),(0.30810776,))]),
            
            'qmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))]),            
            
            'fashionmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))]),     
            
            'imagenetSketch': transforms.Compose([
                    transforms.Resize(size=(28,28)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                         std  = [0.229,0.224,0.225])])}                                        
        return train_transform[self.dataset]

    @property
    def val_transform(self):
        test_transform = {
            'cifar10': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            
            'cifar100': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                ]),
            
            'mnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                ]),
            
            'cinic10': transforms.Compose([
                    transforms.Resize(size=256),
                    transforms.RandomCrop(224),                     
                    transforms.ToTensor(),
                    transforms.Normalize((0.47889522,0.47227842,0.43047404),
                                         (0.24205776,0.23828046,0.25874835))]),
            
            'oxfordIIITPets': transforms.Compose([
                    transforms.Resize(size=32),
                    transforms.CenterCrop(28),                            
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),
                                         (0.229,0.224,0.225))]),
            
            'medmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))]),
            
            'kmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))]),
            
            'emnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.13066062,),(0.30810776,))]),
            
            'qmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,),(0.3081,))]),            

            'fashionmnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))]),     
            
            'imagenetSketch': transforms.Compose([
                    transforms.Resize(size=32),
                    transforms.CenterCrop(28),             
                    transforms.ToTensor(),
                    transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                         std  = (0.229,0.224,0.225))]),
            }
        return test_transform[self.dataset]

    def val_transform2(self):
        test_transform = {
            'isic': transforms.Compose([
                transforms.Resize(size=(28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))]),

            'rfmid': transforms.Compose([
                transforms.Resize(size=(28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
            }
        return test_transform[self.dataset2]

        
    # Official Dataset
    '''
    @property
    def data_cls(self):
        data_cls = {
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'mnist': datasets.MNIST,
            'fashionmnist': datasets.FashionMNIST,
            'kmnist': datasets.KMNIST,
            'qmnist': datasets.QMNIST,
            'emnist': datasets.EMNIST,
            'oxfordIIITPets': datasets.OxfordIIITPet            
        }
        return data_cls[self.dataset]        
    '''

    # torchvision dataset
    '''
    @property
    def train_dataset(self):
        return self.data_cls(self.root, train=True, download=True, transform=self.train_transform)
        #return self.data_cls(self.root, split='trainval',download=True, transform=self.train_transform)

    @property
    def val_dataset(self):
        return self.data_cls(self.root, train=False, download=True, transform=self.val_transform)
        #return self.data_cls(self.root, split='test',download=True, transform=self.train_transform)        

    @property
    def test_dataset(self):
        return self.data_cls(self.root, train = False, download=True, transform=self.val_transform)
        #return self.data_cls(self.root, split='test',download=True, transform=self.train_transform)        
    '''
    
    # EMNIST
    # byclass bymerge balanced letters digits mnist      
    '''
    @property
    def train_dataset(self):
        values = datasets.EMNIST(root='./data/emnist/train',
                                 split="digits", 
                                 train = True,                                                                   
                                 transform=self.train_transform,
                                 download=True)                
        return values
    
    
    @property
    def val_dataset(self):
        values = datasets.EMNIST(root='./data/emnist/valid',
                                 split="digits", 
                                 train = False,                                                                                                    
                                 transform=self.val_transform,
                                 download=True)                
        return values


    @property
    def test_dataset(self):
        values = datasets.EMNIST(root='./data/emnist/test',                                 
                                 split ="digits",                                                                   
                                 train = False, 
                                 transform=self.val_transform,
                                 download=True)                
        return values        
    '''
    
    # QMNIST    
    '''
    @property
    def train_dataset(self):
        values = datasets.QMNIST(root='./data/qmnist/train',                                 
                                 train = True,                                                                   
                                 transform=self.train_transform,
                                 download=True)                
        return values
    
    
    @property
    def val_dataset(self):
        values = datasets.QMNIST(root='./data/qmnist/valid',                                 
                                 train = False,                                                                                                    
                                 transform=self.val_transform,
                                 download=True)                
        return values


    @property
    def test_dataset(self):
        values = datasets.QMNIST(root='./data/qmnist/test',                                                                  
                                 train = False, 
                                 transform=self.val_transform,
                                 download=True)                
        return values        
    '''    
    
    # FashionMNIST
    '''
    @property
    def train_dataset(self):
        values = datasets.FashionMNIST(root='./data/fashionmnist/train',                                                                                                                   
                                 transform=self.train_transform,
                                 download=True)                
        return values
    
    
    @property
    def val_dataset(self):
        values = datasets.FashionMNIST(root='./data/fashionmnist/valid',
                                 transform=self.val_transform,
                                 download=True)                
        return values


    @property
    def test_dataset(self):
        values = datasets.FashionMNIST(root='./data/fashionmnist/test',
                                 transform=self.val_transform,
                                 download=True)                
        return values    
    '''

    # Customized Dataset
    # Dermatoscopic image
    '''
    @property
    def train_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/DermaMNIST', split='train', transform=self.train_transform,
                                download=True, size=28)
        return values

    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/DermaMNIST', split='val', transform=self.val_transform,
                                download=True, size=28)
        return values

    @property
    def test_dataset(self):
        values = datasets.ImageFolder(root='./data/Data/Skin_Imaging/ISIC_2020_Test_JPEG/ISIC_2020_Test_Input',transform=self.val_transform2)
        return values
    '''


    # RetinaMNIST
    '''
    @property
    def train_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/RetinaMNIST', split='train', transform=self.train_transform,
                                download=True, size=28)
        return values

    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/RetinaMNIST', split='val', transform=self.val_transform,
                                download=True, size=28)
        return values

    @property
    def test_dataset(self):
        values = datasets.ImageFolder(root='./data/Data/Retinal_Fundus/A. RFMiD_All_Classes_Dataset/1.Original Images/c.Testing Set',transform=self.val_transform2)
        return values
    '''

    # Cinic10    
    '''
    @property
    def train_dataset(self):
        values = datasets.ImageFolder(root='./data/cinic10/train',transform=self.train_transform)                
        return values
    
    
    @property
    def val_dataset(self):
        values = datasets.ImageFolder(root='./data/cinic10/valid',transform=self.val_transform)                
        return values


    @property
    def test_dataset(self):
        values = datasets.ImageFolder(root='./data/cinic10/test',transform=self.val_transform)                
        return values
    '''
    
    # ImageNetSketch
    '''
    @property
    def train_dataset(self):
        values = datasets.ImageFolder(root='./data/ImageNet-Sketch/sketch',transform=self.train_transform)                
        return values
    
    
    @property
    def val_dataset(self):
        values = datasets.ImageFolder(root='./data/ImageNet-Sketch/sketch',transform=self.val_transform)                
        return values


    @property
    def test_dataset(self):
        values = datasets.ImageFolder(root='./data/ImageNet-Sketch/sketch',transform=self.val_transform)                
        return values
    '''    
            
    # PathMNIST    
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values


    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values    
    '''            

    # BreastMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/BreastMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/BreastMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values


    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/BreastMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values       
    '''

    
    # PathMNIST
    '''
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='train',transform=self.train_transform, download=True )
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='val',transform=self.val_transform, download=True)
        return values
        
    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PathMNIST',split='test',transform=self.val_transform, download=True)
        return values       
    '''

    # OCTMNIST    
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/OCTMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OCTMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values


    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OCTMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values    
    '''    
    
    
    # RetinaMNIST    
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/RetinaMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/RetinaMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/RetinaMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values        
    '''    
    
    
    # BloodMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/BloodMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/BloodMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values
    
    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/BloodMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values            
    '''
    
    # PneumoniaMNIST    
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/PneumoniaMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PneumoniaMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values
    
    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/PneumoniaMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values            
    '''    
    
    
    # ChestMNIST
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/ChestMNIST',split='train',transform=self.train_transform, download=True,size=28)
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/ChestMNIST',split='val',transform=self.val_transform, download=True,size=28)
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/ChestMNIST',split='test',transform=self.val_transform, download=True,size=28)
        return values    


    # DermaMNIST
    '''    
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/DermaMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/DermaMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/DermaMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values        
    '''
    
    # TissueMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/TissueMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/TissueMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/TissueMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values            
    '''

    # OrganAMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/OrganAMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganAMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganAMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values    
    '''
    
    # OrganCMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/OrganCMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganCMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganCMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values    
    '''    
    
    # OrganSMNIST
    '''
    @property
    def train_dataset(self):        
        values = self.DataClass(root='./data/MedMNIST2D/OrganSMNIST',split='train',transform=self.train_transform, download=True, size=28)                
        return values    
    
    @property
    def val_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganSMNIST',split='val',transform=self.val_transform, download=True, size=28)                
        return values

    @property
    def test_dataset(self):
        values = self.DataClass(root='./data/MedMNIST2D/OrganSMNIST',split='test',transform=self.val_transform, download=True, size=28)                
        return values            
    '''    