# This code uses torchvision to download the CIFAR-10 dataset.
# You can also download the dataset manually and extract it to the specified directory.
import torchvision
import torchvision.transforms as transforms

trainset = torchvision.datasets.CIFAR10(
    root='.', 
    train=True, # Train set (50k images)
    download=True,      
    transform=transforms.ToTensor()
)

testset = torchvision.datasets.CIFAR10(
    root='.',
    train=False, # Test set (10k images)
    download=True,
    transform=transforms.ToTensor()
)

