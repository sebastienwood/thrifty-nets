import os
import deit
import argparse
import thrifty_deit
import torch
import torchvision
import torchvision.transforms as transforms
import random
from timm.models import create_model

parser = argparse.ArgumentParser(description="Vincent's Training Routine")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--dataset', type=str, default="CIFAR10", help="CIFAR10, CIFAR100 or ImageNet")
parser.add_argument('--steps', type=int, default=750000)
parser.add_argument('--batch-size', type = int, default=1024)
parser.add_argument('--seed', type = int, default = random.randint(0, 1000000000))
parser.add_argument('--width', type=int, default=64, help="number of feature maps")
parser.add_argument('--dataset-path', type=str, default=os.getenv("DATASETS"))
parser.add_argument('--cifar-resize', type=int, default=32)
parser.add_argument('--label-smoothing', type=float, default=0.1)
parser.add_argument('--adam', action="store_true")
args = parser.parse_args()

def count_parameters(model):
    num_parameters = sum([x.numel() for x in model.parameters()])
    print("{:d} parameters".format(num_parameters))
    return num_parameters

original = create_model('deit_small_patch16_LS', img_size=32)
print('Original')
count_parameters(original)
thrifty = create_model('thrifty_deit_small_patch16_LS', img_size=32)
print('Thriftified')
count_parameters(thrifty)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tvdset = torchvision.datasets.CIFAR10
print(args.dataset_path)
num_classes = 10
train = tvdset(
        root=args.dataset_path,
        train=True,
        download=True,
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
#            torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            transforms.Resize(args.cifar_resize, antialias=True),#, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomErasing(0.1)
        ]))
train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, drop_last=True)

batch = next(iter(train_loader))
original(batch[0])
thrifty(batch[0])