CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/002_cifar10_supervised.yaml --tag 0518/supervised
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/003_cifar10_label_smoothing.yaml --tag 0518/label_smooth
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/004_cifar10_randaugment.yaml --tag 0518/randaugment

CUDA_VISIBLE_DEVICES=3 python main.py --config ./config/005_cifar10_uda.yaml --tag 0518/uda
CUDA_VISIBLE_DEVICES=0 python main.py --config ./config/006_cifar10_supervised_mpl.yaml --tag 0518/supervised_mpl
CUDA_VISIBLE_DEVICES=1 python main.py --config ./config/007_cifar10_randaugment_mpl.yaml --tag 0518/randaugment_mpl
CUDA_VISIBLE_DEVICES=2 python main.py --config ./config/008_cifar10_uda_mpl.yaml --tag 0518/uda_mpl
