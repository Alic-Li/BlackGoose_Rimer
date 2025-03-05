# BlackGoose Rimer: RWKV as a Superior Architecture for Large-Scale Time Series Modeling
![RRFVCM](./back_goose.png)

## 1. Project Overview
#### A general time series forecasting model based on the RWKV_v7 architecture

## 2. Features
#### High training prediction efficiency, low video memory usage, high prediction accuracy
#### The prediction accuracy is higher than that of the Tsinghua Timer project, and the training and prediction speed is four times that of the Timer project  which based on the Transformer architecture.

## 3. Environment Setup
### 3.1 System Requirements

This code run seccessfully on the following GPU environment:
* NVIDIA GeForce RTX 4060laptop 8G (Ubuntu 24.04 Python 3.12 CUDA 12.4)
* AMD Radeon Pro W7900 48G (Ubuntu 24.04 Python 3.12 ROCm 6.3)
* AMD Radeon RX 6750xt 12G (Ubuntu 24.04 Python 3.12 ROCm 6.3)

### 3.2 Install Dependencies

```bash
git clone https://github.com/Alic-Li/RWKV_V7_Black_Goose_Sequence_Forecasting.git
cd RWKV_V7_Black_Goose_Sequence_Forecasting
pip install torch torchvision torchaudio tqdm numpy rwkv-fla scikit-learn joblib matplotlib pandas 
```

### 3.3 Data Preparation

Datasets in the path of ```./dataset/```

## 4. Usage
### 4.1 Training
```bash
python ./train.py 
```
### 4.2 Testing & Evaluation
```bash
python ./pridict.py
```
### 4.3 Testing & Evaluation with plot
```bash
python ./pridict_with_plot.py
```
## Thanks
- RWKV_V7: https://github.com/BlinkDL/RWKV-LM
- This code is based on the https://github.com/thuml/Large-Time-Series-Model
