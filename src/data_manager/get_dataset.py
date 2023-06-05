import torch
from torch.utils.data import TensorDataset

import torchvision.datasets as datasets
from torchvision.datasets import MNIST, CIFAR10, Food101, CelebA
from data_manager.colored_mnist import ColoredMNIST
from data_manager.stacked_mnist import StackedMNIST
from data_manager.stacked_mnist_rand_pos import StackedMNISTRandPos
from data_manager.left_or_right_mnist import LeftOrRightMNIST
from torchvision import transforms
import torchvision.transforms.functional as tvf
import sklearn.datasets as skd

import numpy as np


# Messy code to get the dataset
# Works fine

transform_dict = {
    "small_image_dataset": transforms.Compose([
        transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "mnist": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "mnist_half_dimmed": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.Lambda(lambda x: x*0.2),
        ], p=0.5),
        transforms.Normalize((0.5), (0.5))
    ]),
    "stacked_mnist": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "stacked_mnist_rand_pos": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "colored_mnist": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "left_mnist": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "right_mnist": transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "food101": transforms.Compose([
        transforms.CenterCrop((376, 376)),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "food128": transforms.Compose([
        transforms.CenterCrop((376, 376)),
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celeb256": transforms.Compose([
        transforms.CenterCrop((256, 256)),
        # transforms.Resize((128, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celeba": transforms.Compose([
        transforms.CenterCrop((178, 178)),
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celebahq256": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celebahq512": transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celebacropped": transforms.Compose([
        transforms.Lambda(lambda x: tvf.crop(x, 121-64, 89-64, 128, 128)),
        # transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "celeba64cropped": transforms.Compose([
        transforms.Lambda(lambda x: tvf.crop(x, 121-64, 89-64, 128, 128)),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "afhq512": transforms.Compose([
        # transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "afhq256": transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "afhq128": transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "theoffice256": transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "ffhq512": transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "pokemoncards64": transforms.Compose([
        # crop the top of the image to a square
        transforms.Lambda(lambda x: x.crop((0, 0, x.size[0], x.size[0]))),
        # Remove some of the border
        # transforms.Lambda(lambda x: x.crop((0.01*x.size[0], 0.01*x.size[0], 0.99*x.size[0], 0.99*x.size[0]))),
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ]),
    "pokemoncards128": transforms.Compose([
        # crop the top of the image to a square
        transforms.Lambda(lambda x: x.crop((0, 0, x.size[0], x.size[0]))),
        # Remove some of the border
        # transforms.Lambda(lambda x: x.crop((0.01*x.size[0], 0.01*x.size[0], 0.99*x.size[0], 0.99*x.size[0]))),
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "pokemoncards256": transforms.Compose([
        # crop the top of the image to a square
        transforms.Lambda(lambda x: x.crop((0, 0, x.size[0], x.size[0]))),
        # Remove some of the border
        # transforms.Lambda(lambda x: x.crop((0.01*x.size[0], 0.01*x.size[0], 0.99*x.size[0], 0.99*x.size[0]))),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

def get_dataset(name, transform=None, **kwargs):
    if name not in transform_dict:
        print("No transform found for dataset: ", name)
    else:
        transform = transform_dict[name] if transform is None else transform
    
    if name != "gaussian_mixture":
        kwargs.pop("mus", None)
        kwargs.pop("sigmas", None)

    # Image datasets
    if name in ["mnist", "mnist_half_dimmed", "mnist_original_size", "left_mnist", "right_mnist"]:
        if name in ["left_mnist", "right_mnist"]:
            position = 0 if name == "left_mnist" else 1
            return LeftOrRightMNIST(position=position, transform=transform, **kwargs)
        return MNIST(transform=transform, **kwargs)
    elif name == "stacked_mnist":
        return StackedMNIST(transform=transform, **kwargs)
    elif name == "stacked_mnist_rand_pos":
        return StackedMNISTRandPos(transform=transform, **kwargs)
    elif name == "colored_mnist":
        return ColoredMNIST(transform=transform, **kwargs)
    elif name == "cifar10":
        return CIFAR10(transform=transform, **kwargs),
    elif name in ["food101", "food128"]:
        kwargs.pop("train", None)
        return Food101(transform=transform, **kwargs)
    elif name in ["celeba256", "celeba", "celebacropped", "celeba64cropped"]:
        if "train" in kwargs:
            kwargs.pop("train", None)
            kwargs["split"] = "train"
        else:
            kwargs["split"] = "all"
        return CelebA(transform=transform, **kwargs)
    elif name in ["celebahq256"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/celebahq256", transform=transform, **kwargs_pruned)

    elif name in ["celebahq512"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/celebahq512", transform=transform, **kwargs_pruned)

    elif name in ["afhq128", "afhq256", "afhq512"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/afhq", transform=transform, **kwargs_pruned)
    elif name in ["theoffice256"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/theoffice", transform=transform, **kwargs_pruned)
    
    elif name in ["ffhq512"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/ffhq", transform=transform, **kwargs_pruned)
    
    elif name in ["pokemoncards64", "pokemoncards128", "pokemoncards256"]:
        kwargs_pruned = kwargs.copy()
        kwargs_pruned.pop("root", None)
        kwargs_pruned.pop("train", None)
        kwargs_pruned.pop("download", None)

        return datasets.ImageFolder(root=f"{kwargs['root']}/pokemoncards", transform=transform, **kwargs_pruned)

    elif name == "small_image_dataset":
        small_image_dataset = MNIST(transform=transform, **kwargs)
        small_image_dataset.data = small_image_dataset.data[:1000]
        return small_image_dataset

    
    # 1D shape datasets
    elif name == "gaussian_mixture":
        n = 4000000
        mus, sigmas = kwargs["mus"], kwargs["sigmas"]
        print("mus: ", mus)
        print("sigmas: ", sigmas)
        dims = len(mus[0])
        num_gaussians = len(mus)
        if dims == 1:
            dataset = torch.cat([torch.normal(mean=m[0], std=s[0][0], size=(n//num_gaussians, 1)) for m, s in zip(mus, sigmas)], dim=0)

        elif dims == 2:
            from torch.distributions.multivariate_normal import MultivariateNormal
            mus = torch.tensor(mus).float()
            sigmas = torch.tensor(sigmas).float()
            dataset = torch.cat([MultivariateNormal(m, c).sample((n//num_gaussians,)) for m, c in zip(mus, sigmas)], dim=0)
        dataset = dataset[torch.randperm(dataset.shape[0])]
        return TensorDataset(dataset)

    elif name == "laplace":
        n = 3000000
        from torch.distributions import Laplace
        m = Laplace(torch.tensor([0.0]), torch.tensor([torch.sqrt(torch.tensor([2.0]))]))
        dataset = m.sample((n,))
        return TensorDataset(dataset)

    elif name == "uniform":
        n = 3000000
        # Centered around 0 with width 4
        dataset = torch.rand((n, 1)) * 4 - 2
        return TensorDataset(dataset)

    elif name == "two_boxes":
        n = 4000000
        dataset = torch.cat([(torch.rand((n//2, 2))-0.5) * 2 - torch.tensor([5, 0]), (torch.rand((n//2, 2))-0.5) * 2 + torch.tensor([5, 0])], dim=0)
        dataset = dataset[torch.randperm(dataset.shape[0])]
        return TensorDataset(dataset)

    elif name == "dirac":
        n = 3000000
        dataset = torch.cat([-torch.ones((n, 1)) * 2, torch.ones((n, 1))*2], dim=0)
        dataset = dataset[torch.randperm(dataset.shape[0])]
        return TensorDataset(dataset)


    elif name == "swiss_roll_2d":
        data = skd.make_swiss_roll(n_samples=10000000, noise=0.45)[0] 
        data = torch.from_numpy(data).float()
        data = data[:, [0, 2]] * 0.15
        return TensorDataset(data)
    
    elif name == "s_curve_2d":
        data = skd.make_s_curve(n_samples=10000000, noise=0.15)[0]
        data = torch.from_numpy(data).float()
        data = data[:, [0, 2]] 
        return TensorDataset(data)
    elif name == "s_curve_2d_transformed":
        data = skd.make_s_curve(n_samples=10000000, noise=0.0)[0]
        data = data[:, [0, 2]]
        data[:, 1] = data[:, 1] * 0.5
        data = data * 1.9
        data += np.random.normal(0, 0.15, size=data.shape)
        data = torch.from_numpy(data).float()
        return TensorDataset(data)

    elif name == "circles":
        data = skd.make_circles(n_samples=4000000, noise=0.05, factor=0.5, random_state=0)[0]
        data = torch.from_numpy(data).float()
        data = data * 10
        return TensorDataset(data)

    else:
        raise NotImplementedError


