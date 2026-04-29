# KoHD

# 1. Installation
This repository is based on PyTorch 2.1.0, CUDA 12.1 and Python 3.11
```bash
conda create -n KoHD python=3.11
conda activate KoHD
pip install -r requirements.txt
```
# 2. Dataset
You can refer to (https://github.com/HUANGLIZI/LViT) to download the dataset.
```bash
├── ./datasets
│   ├── Covid19
│   │   ├── Train_Folder
│   │   ├── Val_Folder
│   │   ├── Test_Folder
│   │   ├── Train_text.xlsx
│   │   ├── Val_text.xlsx
│   │   ├── Test_text.xlsx
│   │   ├── Test_text_split.xlsx
│   ├── MosMedplus
│   │   ├── Train_Folder
│   │   ├── Val_Folder
│   │   ├── Test_Folder
│   │   ├── Train_text.xlsx
│   │   ├── Val_text.xlsx
│   │   ├── Test_text.xlsx
│   │   ├── Test_text_split.xlsx
│   ├── MoNuSeg
│   │   ├── Train_Folder
│   │   ├── Val_Folder
│   │   ├── Test_Folder
│   │   ├── Train_text.xlsx
│   │   ├── Val_text.xlsx
│   │   ├── Test_text.xlsx
│   │   ├── Test_text_split.xlsx
│   ├── Kvasir
│   │   ├── Train_Folder
│   │   ├── Val_Folder
│   │   ├── Test_Folder
│   │   ├── Train_text.xlsx
│   │   ├── Val_text.xlsx
│   │   ├── Test_text.xlsx
│   │   ├── Test_text_split.xlsx
```
# 3. Usage
To train a model,

```bash
python ./code/train_model.py
```
To test a model,
```bash
python ./code/test.py  
```
# Acknowledgements
Our code is largely based on (https://github.com/HUANGLIZI/LViT). Thanks for these authors for their valuable work, hope our work can also contribute to related research.
