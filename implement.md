# Implementation Details

## Baselines
![baseline](./figures/00_baseline.gif){: width="60%" height="60%"}
### Supervised
![supervised](./figures/01_supervised.png){: width="60%" height="60%"}
### Label Smoothing
![label_smoothing](./figures/02_label_smoothing.png){: width="60%" height="60%"}
### RandAugment
![randaugment](./figures/03_randaugment.png){: width="60%" height="60%"}
### UDA
![uda](./figures/04_uda.png){: width="60%" height="60%"}

## Baseline + MPL
![baseline_mpl](./figures/10_baseline.gif){: width="60%" height="60%"}
### Supervised + MPL
![supervised_mpl](./figures/11_supervised_mpl.png){: width="60%" height="60%"}
### RandAugment + MPL
![randaugment_mpl](./figures/12_randaugment_mpl.png){: width="60%" height="60%"}
### UDA + MPL
![uda_mpl](./figure/13_uda_mpl.png){: width="60%" height="60%"}

## Two Phases Traing of MPL
![two_phases](./figures/20_two_phases.gif){: width="60%" height="60%"}
### Phase 1: Updating Student (Supervised + MLP)
![two_phases](./figures/21_two_phases_1.png){: width="60%" height="60%"}
### Phase 2-1: Updating Teacher (Supervised + MLP)
![two_phases](./figures/21_two_phases_2.png){: width="60%" height="60%"}
### Phase 2-2: Updating Teacher (UDA + MLP)
![two_phases](./figures/22_two_phases_2_uda.png){: width="60%" height="60%"}
