# Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer [WACV 2023]
This is the Official PyTorch implemention of our WACV2023 paper **Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer**

[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Chen_Federated_Domain_Generalization_for_Image_Recognition_via_Cross-Client_Style_Transfer_WACV_2023_paper.html) |  [Supp](https://openaccess.thecvf.com/content/WACV2023/supplemental/Chen_Federated_Domain_Generalization_WACV_2023_supplemental.pdf) | [Arxiv](https://arxiv.org/abs/2210.00912) | [Project Page](https://chenjunming.ml/proj/CCST/)
## Usage
### Setup
**pip**
See the `requirements.txt` for environment configuration. 
```bash
pip install -r requirements.txt
```

### Datasets
- PACS: Please download according to the official github repo of JiGen: https://github.com/fmcarlucci/JigenDG.
- OfficeHome: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
- Camelyon17: https://camelyon17.grand-challenge.org/Data/

Remenber to change the path of images in txt files under ```./data/txt_lists/``` as yours.

### Cross-Client Style Transfer (CCST)
Download [decoder.pth](https://drive.google.com/file/d/1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr/view?usp=sharing) / [vgg_normalized.pth](https://drive.google.com/file/d/1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU/view?usp=sharing) and put them under ```./style_transfer/AdaIN/models/```.

To perform CCST in the mode of Overall (K=3) for PACS, you can run the following:
```bash
cd style_transfer/AdaIN

# Overall Style computation
CUDA_VISIBLE_DEVICES=0 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target art_painting --batch 32 &
CUDA_VISIBLE_DEVICES=1 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target cartoon --batch 32 &
CUDA_VISIBLE_DEVICES=2 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target photo --batch 32 &
CUDA_VISIBLE_DEVICES=3 python mean_std_computation_effcientMem.py --dataset pacs --image_size 512 --target sketch --batch 32

# Overall Style Transfer
CUDA_VISIBLE_DEVICES=0 python CCST_OverallStyleTransfer.py --dataset pacs --target art_painting --batch 6 --image_size 512 &
CUDA_VISIBLE_DEVICES=1 python CCST_OverallStyleTransfer.py --dataset pacs --target cartoon --batch 6 --image_size 512 &
CUDA_VISIBLE_DEVICES=2 python CCST_OverallStyleTransfer.py --dataset pacs --target photo --batch 6 --image_size 512 &
CUDA_VISIBLE_DEVICES=3 python CCST_OverallStyleTransfer.py --dataset pacs --target sketch --batch 6 --image_size 512 &

# Reorgnize data from all_style_transferred_Overall
## cd data/, change the path in reorganize_dataset.py to your own
python reorganize_dataset.py --dataset PACS --mode Overall --target art_painting &
python reorganize_dataset.py --dataset PACS --mode Overall --target cartoon &
python reorganize_dataset.py --dataset PACS --mode Overall --target photo &
python reorganize_dataset.py --dataset PACS --mode Overall --target sketch 
```

For CCST single style mode:
```bash 
cd style_transfer/AdaIN

# Single Style Computation and Transfer
CUDA_VISIBLE_DEVICES=1 python CCST_SingleStyleTransfer.py --dataset pacs --target art_painting --batch 32 --image_size 512 &
CUDA_VISIBLE_DEVICES=2 python CCST_SingleStyleTransfer.py --dataset pacs --target cartoon --batch 6 --image_size 512 &
CUDA_VISIBLE_DEVICES=3 python CCST_SingleStyleTransfer.py --dataset pacs --target photo --batch 6 --image_size 512 &
CUDA_VISIBLE_DEVICES=0 python CCST_SingleStyleTransfer.py --dataset pacs --target sketch --batch 6 --image_size 512 &

# Reorgnize data from all_style_transferred_Single
## cd data/, change the path in reorganize_dataset.py to your own
python reorganize_dataset.py --dataset PACS --mode Single --target art_painting &
python reorganize_dataset.py --dataset PACS --mode Single --target cartoon &
python reorganize_dataset.py --dataset PACS --mode Single --target photo &
python reorganize_dataset.py --dataset PACS --mode Single --target sketch 
```

Then, generate the dataset lists to be loaded during traing:

```bash
cd data
## PACS, Overall, K=3
python data_list_generator.py --dataset PACS --target art_painting --mode overall --style adain --K 3 &
python data_list_generator.py --dataset PACS --target cartoon --mode overall --style adain --K 3 &
python data_list_generator.py --dataset PACS --target photo --mode overall --style adain --K 3 &
python data_list_generator.py --dataset PACS --target sketch --mode overall --style adain --K 3 &
```

### Train

- **--fusion_mode** :specify style transfer mode, includes single and overall modes.

  - For PACS and OfficeHome

    - 'adain-single-K1': Single(K=1)
    - 'adain-single-K2': Single(K=2)
    - 'adain-single-K3': Single(K=3)
    - 'adain-overall-K1': Overall (K=1)
    - 'adain-overall-K2': Overall (K=2)
    - 'adain-overall-K3': Overall (K=3)

  - For Camelyon17

    - 'adain-single-K4': Single(K=4)
    - 'adain-overall-K4': Overall (K=4)

- **--dg_method**: Use other Domain Generalization methods under the Federated Learning (FedAvg) setting. Choices: ['no_DG', 'RSC', 'Jigsaw', 'MixStyle', 'feddg'].

#### PACS
Please using following commands to train a model with photo as target using ResNet50 in overall mode with K=3.
```bash
python fed_run.py --mode fedavg  --fusion_mode adain-overall-K3 --source art_painting cartoon sketch --target photo  --random_horiz_flip 0.5 --n_classes 7 --network resnet50 --lr 0.001 --image_size 222 --batch 64 --log
```
#### OfficeHome
Please using following commands to train a model with art as target using ResNet18 in overall mode with K=3.
```bash
python fed_run.py --mode fedavg --dataset officehome --fusion_mode adain-overall-K3 --source clipart product real_world --target art  --random_horiz_flip 0.5 --n_classes 65 --network resnet18 --lr 0.001 --image_size 222 --batch 32 --log
```
#### Camelyon17
Please using following commands to train a model with hospital5 as target using DenseNet121 in overall mode with K=4.
```bash
python fed_run.py --mode fedavg --dataset camelyon17 --fusion_mode adain-overall-K4 --source hospital1 hospital2 hospital3 hospital4  --target hospital5  --random_horiz_flip 0.5 --n_classes 2 --network densenet --lr 0.001 --image_size 96 --batch 32 --log --iters 200

```

You can find more running commands in federated/run.sh


### Test
#### PACS
Please using following commands to test a model with photo as target using ResNet50 in overall mode with K=3.
Note that the checkpoint path has to be specified before test.
```bash
python fed_run.py --mode fedavg  --fusion_mode adain-overall-K3 --source art_painting cartoon sketch --target photo --n_classes 7 --network resnet50 --lr 0.001 --image_size 222 --batch 64 --test
```
#### OfficeHome
Please using following commands to test a model with art as target using ResNet18 in overall mode with K=3.
Note that the checkpoint path has to be specified before test.

```bash
python fed_run.py --mode fedavg --dataset officehome --fusion_mode adain-overall-K3 --source clipart product real_world --target art  --n_classes 65 --network resnet18 --lr 0.001 --image_size 222 --batch 32 --test
```

#### Camelyon17
Please using following commands to test a model with hospital5 as target using DenseNet121 in overall mode with K=4.
```bash
python fed_run.py --mode fedavg --dataset camelyon17 --fusion_mode adain-overall-K4 --source hospital1 hospital2 hospital3 hospital4  --target hospital5  --random_horiz_flip 0.5 --n_classes 2 --network densenet --lr 0.001 --image_size 96 --batch 32 --test

```

## BibTeX
```
@InProceedings{Chen_2023_WACV,
    author    = {Chen, Junming and Jiang, Meirui and Dou, Qi and Chen, Qifeng},
    title     = {Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {361-370}
}
```