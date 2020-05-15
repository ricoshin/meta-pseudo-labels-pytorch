# Implementation Details

## Baselines
![baseline](./figures/00_baseline.gif =100)
### Supervised
![supervised](./figures/01_supervised.png =150)
### Label Smoothing
![label_smoothing](./figures/02_label_smoothing.png =200)
### RandAugment
![randaugment](./figures/03_randaugment.png)
### UDA
![uda](./figures/04_uda.png)

## Baseline + MPL
![baseline_mpl](./figures/10_baseline_mpl.gif)
### Supervised + MPL
![supervised_mpl](./figures/11_supervised_mpl.png)
### RandAugment + MPL
![randaugment_mpl](./figures/12_randaugment_mpl.png)
### UDA + MPL
![uda_mpl](./figures/13_uda_mpl.png)

## Two Phases Traing of MPL
![two_phases](./figures/20_two_phases.gif)
### Phase 1: Updating Student (Supervised + MLP)
![two_phases_1](./figures/21_two_phases_1.png)
### Phase 2-1: Updating Teacher (Supervised + MLP)
![two_phases_2](./figures/21_two_phases_2.png)
### Phase 2-2: Updating Teacher (UDA + MLP)
![two_phases_w_uda](./figures/22_two_phases_2_uda.png)
