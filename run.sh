python main.py --autotag --config ./config/002_cifar10_supervised.yaml (on)
python main.py --autotag --config ./config/003_cifar10_label_smoothing.yaml
python main.py --autotag --config ./config/004_cifar10_randaugment.yaml
python main.py --autotag --config ./config/040_cifar10_uda_2.yaml (on)
python main.py --autotag --config ./config/006_cifar10_supervised_mpl.yaml
python main.py --autotag --config ./config/007_cifar10_randaugment_mpl.yaml
python main.py --autotag --config ./config/008_cifar10_uda_mpl.yaml

python tune.py  --autotag --config ./config/010_cifar10_supervised.yaml (fin)
python tune.py  --autotag --config ./config/020_cifar10_label_smoothing.yaml
python tune.py  --autotag --config ./config/030_cifar10_randaugment.yaml
python tune.py  --autotag --config ./config/040_cifar10_uda_2.yaml (fin)
python tune.py  --autotag --config ./config/060_cifar10_supervised_mpl.yaml (on)
python tune.py  --autotag --config ./config/070_cifar10_randaugment_mpl.yaml
python tune.py  --autotag --config ./config/080_cifar10_uda_mpl.yaml (on)
