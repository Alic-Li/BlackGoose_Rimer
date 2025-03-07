# BlackGoose Rimer: RWKV as a Superior Architecture for Large-Scale Time Series Modeling 
![RRFVCM](./back_goose.png)

## 1. Project Overview 🫣
#### A general time series forecasting model based on the RWKV_v7 architecture

## 2. Features 🐦
#### High training prediction efficiency, low video memory usage, high prediction accuracy
#### The prediction accuracy is higher than that of the Tsinghua Timer project, and the training and prediction speed is four times that of the Timer project  which based on the Transformer architecture.

## 3. Benchmarks 🚀
### ELC test dataset ⚡
| Model             | RMSE          | MAE           | MAPE   🥲 |  R^2       | 
| :-----            |:----:         |:----:         |:----:     |:----:      |
| Rimer_RWKV_v7     |```0.2409```   |```0.0814```   |0.81%      |```0.9991```|
| Timer_Transformer |0.6488         |0.2127         |```0.61%```|0.9755      |

### ETTH test dataset ⚡
| Model             | RMSE  🥲      | MAE   🥲      | MAPE      |  R^2       | 
| :-----            |:----:         |:----:         |:----:     |:----:      |
| Rimer_RWKV_v7     |0.1793         |0.0298         |```0.42%```|```0.9660```|
| Timer_Transformer |```0.0055```   |```0.0015```   |23.95%     |0.8901      |

### Traffic test dataset 🚥
| Model             | RMSE          | MAE           | MAPE      |  R^2       | 
| :-----            |:----:         |:----:         |:----:     |:----:      |
| Rimer_RWKV_v7     |```0.0025```   |```0.0006```   |```4.01%```|```0.9838```|
| Timer_Transformer |0.0055         |0.0015         |19.94%     |0.8955      |

### Weather test dataset 🌦️
| Model             | RMSE          | MAE           | MAPE      |  R^2       | 
| :-----            |:----:         |:----:         |:----:     |:----:      |
| Rimer_RWKV_v7     |```5.4311```   |```1.3621```   |```0.34%```|```0.8794```|
| Timer_Transformer |6.1765         |3.6839         |0.88%      |0.8411      |

## 4. Environment Setup
### 4.1 System Requirements 😜

This code run seccessfully on the following GPU environment:
* NVIDIA GeForce RTX 4060laptop 8G (Ubuntu 24.04 Python 3.12 CUDA 12.4)
* AMD Radeon Pro W7900 48G (Ubuntu 24.04 Python 3.12 ROCm 6.3)
* AMD Radeon RX 6750xt 12G (Ubuntu 24.04 Python 3.12 ROCm 6.3)

### 4.2 Install Dependencies 

```bash
git clone https://github.com/Alic-Li/RWKV_V7_Black_Goose_Sequence_Forecasting.git
cd RWKV_V7_Black_Goose_Sequence_Forecasting
pip install torch torchvision torchaudio tqdm numpy rwkv-fla scikit-learn joblib matplotlib pandas 
```

### 4.3 Data Preparation 🤗

Datasets in the path of ```./dataset/```

## 5. Usage 
### 5.1 Training 🔥
```bash
python ./train.py 
```
### 5.2 Testing & Evaluation with plot 🤯
```bash
python ./pridict_with_plot.py
```
## Thanks 🫡
- RWKV_V7: https://github.com/BlinkDL/RWKV-LM
- This code is based on the https://github.com/thuml/Large-Time-Series-Model
