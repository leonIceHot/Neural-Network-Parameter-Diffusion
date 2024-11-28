import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from .base import DataBase
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
        self.dataset = getattr(self.cfg, 'dataset', 'cifar10')        
        
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
                    transforms.Resize(size=256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),                                   
                    transforms.ToTensor(),
                    transforms.Normalize((0.47889522,0.47227842,0.43047404),
                                         (0.24205776,0.23828046,0.25874835))]),
            
            'OxfordIIITPet': transforms.Compose([
                    #transforms.Resize(size=256),
                    #transforms.CenterCrop(224),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),                                   
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),
                                         (0.229,0.224,0.225))])
            }                                        
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
                    transforms.RandomHorizontalFlip(),    
                    transforms.ToTensor(),
                    transforms.Normalize((0.47889522,0.47227842,0.43047404),
                                         (0.24205776,0.23828046,0.25874835))]),
            
            'OxfordIIITPet': transforms.Compose([
                    #transforms.Resize(size=256),
                    #transforms.CenterCrop(224),        
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406),
                                         (0.229,0.224,0.225))])            
            }
        
        return test_transform[self.dataset]

        
    # Official Dataset        
    @property
    def data_cls(self):
        data_cls = {
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'mnist': datasets.MNIST,
            'OxfordIIITPet': datasets.OxfordIIITPet            
        }
        return data_cls[self.dataset]        
    
    @property
    def train_dataset(self):
        return self.data_cls(self.root, download=True, transform=self.train_transform)        

    @property
    def val_dataset(self):
        return self.data_cls(self.root, download=True, transform=self.val_transform)        

    @property
    def test_dataset(self):
        return self.data_cls(self.root, download=True, transform=self.val_transform)
    
    
    
    # Customed Dataset                  
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