CLEAN_ROOT_DICT = {
'cifar10': {
    'resnet': '/kaggle/input/clean-resnet18-120models-dataset',
    'preact': '/kaggle/input/cleanset-preact',
},
'cifar100': {
    'resnet': '/kaggle/input/cifar100-renset18-all-models/models/clean',
    'preact': '/kaggle/input/cifar100-preactresnet18-allmodels/clean',
},
'mnist': {
    'resnet': '/kaggle/input/clean-testset-resnet-mnist/models',
    'preact': '/kaggle/input/clean-preactresnet18-mnist-120models-dataset',
},
'gtsrb': {
    'resnet': '/kaggle/input/gtsrb-renset18-all-models/models/clean',
    'preact': '/kaggle/input/gtsrb-preactresnet18-cleans-bads/clean',
},
'celeba': {
    'resnet': '',
    'preact': '',
}
}

BAD_ROOT_DICT = {
'cifar10': {
    'resnet': '/kaggle/input/backdoored-resnet18-120models-6attack-dataset',
    'preact': '/kaggle/input/badset-preact',
    },
'mnist': {
    'resnet': '/kaggle/input/bad-testset-resnet-mnist/models',
    'preact': '/kaggle/input/backdoored-preactresnet18-mnist-100models-5attack',
},
'cifar100': {
    'resnet': '/kaggle/input/cifar100-renset18-all-models/models',
    'preact': '/kaggle/input/cifar100-preactresnet18-allmodels',
},
'gtsrb': {
    'resnet': '/kaggle/input/gtsrb-renset18-all-models/models',
    'preact': '/kaggle/input/gtsrb-preactresnet18-cleans-bads',
},
'celeba': {
    'resnet': '',
    'preact': '',
}
}

# Number of classes
num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'fmnist': 10,
    'gtsrb': 43,
    'celeba': 8,
    'trojai': 5,
}

# Normalizations
NORM_MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4865, 0.4409),
    'mnist': (0.5, 0.5, 0.5),
    'gtsrb': (0, 0, 0),
    'celeba': (0, 0, 0),
}

NORM_STD = {
    'cifar10': (0.247, 0.243, 0.261),
    'cifar100': (0.2673, 0.2564, 0.2762),
    'mnist': (0.5, 0.5, 0.5),
    'gtsrb': (1, 1, 1),
    'celeba': (1, 1, 1),
}

