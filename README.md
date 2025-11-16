# Enhancing Speaker Recognition Robustness with Scalable Deep Learning Models and MFCC Features

This repository contains the implementation for a robust speaker recognition system that integrates **Mel-Frequency Cepstral Coefficients (MFCCs)** with three scalable deep learning architectures: **Feed Forward Neural Network (FFNN)**, **Forward Cascade Back Propagation (FCBP)**, and **Elman Propagation Neural Network (EPNN)**.

<img width="976" height="1114" alt="image" src="https://github.com/user-attachments/assets/4e3799bd-1bf5-4441-9b9d-adb89b9b41dc" />


[![Paper](https://img.shields.io/badge/📄-Published%20Paper-2E86AB?style=for-the-badge)](https://eastpublication.com/index.php/ejcs/article/view/185)
[![DOI](https://img.shields.io/badge/DOI-10.63496/ejcs.Vol1.Iss5.185-1A6FDF?style=for-the-badge&logo=doi&logoColor=white)](https://doi.org/10.63496/ejcs.Vol1.Iss5.185)

## Objective

To develop a robust, high-performance speaker recognition system capable of handling real-world acoustic challenges by integrating:
* **Mel-Frequency Cepstral Coefficients (MFCCs)** for capturing speaker-specific spectral characteristics.
* **Three Neural Architectures (FFNN, FCBP, EPNN)** for comprehensive performance evaluation across diverse conditions.
* **Multi-Dataset Validation** to ensure generalization and robustness in noisy environments.

## Key Contributions

* **Multi-Architecture Analysis:** Comprehensive evaluation of FFNN, FCBP, and EPNN models for speaker recognition under varying acoustic conditions.
* **Robust Feature Engineering:** Implementation of 40-dimensional MFCCs with delta and delta-delta coefficients for enhanced temporal representation.
* **Cross-Dataset Validation:** Rigorously evaluated on three heterogeneous speech databases (**SLR70 Nigerian English, Google Crowdsourced Nigerian English, VoxCeleb2**) to ensure generalization.
* **Real-World Applicability:** Focus on noisy environments and diverse speaker demographics to ensure practical deployment viability.
* **Comprehensive Evaluation:** Performance assessed using multiple metrics: Accuracy, Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Methodology

### 1. Datasets
<img width="133" height="85" alt="image" src="https://github.com/user-attachments/assets/3e9d7fc3-d4b8-4671-930f-50d55c22decb" />

* **SLR70 Nigerian English:** 5,000 utterances (~20 hours), 16 kHz, gender-balanced.
* **Google Crowdsourced Nigerian English:** 7,000 utterances from 1,200 speakers, diverse demographics.
* **VoxCeleb2:** 10,000 utterances subset, celebrity interviews from YouTube, real-world conditions.

### 2. Preprocessing Pipeline
<img width="160" height="152" alt="image" src="https://github.com/user-attachments/assets/561d9858-6057-42f7-b844-110e467bf293" />

* Audio resampling to 16 kHz, amplitude normalization
* Silence trimming using energy threshold (-30 dB)
* Background noise reduction using Butterworth band-pass filter (300-3,400 Hz)

### 3. Feature Extraction

* 40-dimensional MFCC extraction with delta and delta-delta coefficients
* Frame size = 25 ms, hop size = 10 ms
* 120 features per frame including derivatives

### 4. Model Architectures
<img width="157" height="148" alt="image" src="https://github.com/user-attachments/assets/3c71172c-49b4-49e7-89de-fde425fcd991" />

* **FFNN:** 128 neurons, ReLU activation, 0.3 dropout, Adam optimizer (lr=0.001)
* **FCBP:** Cascaded 64→32 neurons, layer-wise training, 0.2 dropout
* **EPNN:** 64 Elman RNN neurons, Tanh activation, BPTT training, 0.3 dropout

### 5. Training Configuration
* 60% training, 30% validation, 10% testing split
* 200 epochs, fixed random seed (42)
* NVIDIA RTX 3090 GPU, TensorFlow 2.12, Python 3.10

## Results

### Performance Comparison


The proposed **EPNN model** demonstrated superior performance across all evaluation metrics:

| Dataset | Model | Accuracy | MSE | MAE | RMSE |
|---------|-------|----------|-----|-----|------|
| **Data 1** | FFNN | 75.75% | 1.198 | 0.511 | 1.094 |
| **Data 1** | FCBP | 70.22% | 0.779 | 0.489 | 0.882 |
| **Data 1** | EPNN | 68.18% | 0.519 | 0.396 | 0.720 |

### Error Metrics Visualization


The EPNN model consistently achieved the lowest error rates across all datasets, demonstrating its robustness in handling temporal dependencies in speech data.

## Performance Analysis

<img width="301" height="188" alt="image" src="https://github.com/user-attachments/assets/faa016f1-5748-4ba9-a9f5-4dedd76fb703" />


* **FFNN:** Best performance in clean conditions (75.75% accuracy) but poor generalization in noisy environments.
* **FCBP:** Stable performance across datasets with consistent error reduction through cascaded training.
* **EPNN:** Superior temporal modeling capabilities, achieving the lowest error rates and best performance in noisy conditions.

## Comparative Analysis



When compared with recent speaker recognition studies, the proposed framework stands out by:
* Providing **comprehensive multi-architecture evaluation** across three distinct neural network paradigms.
* Demonstrating **excellent cross-dataset generalization** from controlled to real-world conditions.
* Achieving **robust performance in noisy environments** through temporal modeling.
* Maintaining **computational efficiency** while ensuring real-world applicability.

## Conclusion & Future Work

<img width="309" height="181" alt="image" src="https://github.com/user-attachments/assets/609c0abb-d499-4172-91da-33b50ce6dec7" />


The **MFCC-based deep learning framework** provides an optimal balance between predictive accuracy, computational efficiency, and real-world robustness for speaker recognition.

**Future work will focus on:**
* Integration of **attention mechanisms** for enhanced temporal modeling.
* Development of **hybrid architectures** combining recurrent and convolutional networks.
* Expansion to **multi-modal speaker recognition** (audio-visual integration).
* **Real-time deployment** optimization for edge devices and mobile applications.
* **Adversarial robustness** testing and enhancement for security-critical applications.

---

## Author

<div align="center">

**Dr. Yasir Hussein Shakir**  
*AI Research Scientist | Artificial Intelligence*

> **Note:** If you encounter any issues with this code, please don't hesitate to contact me.

## Contact Information

<div align="center">

| Platform | Address | Badge |
|----------|---------|-------|
| **🏫 Uniten** | `pe20911@uniten.edu.my` | ![Academic](https://img.shields.io/badge/%F0%9F%93%A7_Academic-00A2FF?style=flat-square) |
| **📮 Yahoo** | `yasserhesseinshakir@yahoo.com` | ![Personal](https://img.shields.io/badge/%F0%9F%93%A8_Personal-720E9E?style=flat-square) |
| **📚 Google Scholar** | [`Yasir Hussein Shakir`](https://scholar.google.com/citations?user=37iNJq0AAAAJ&hl=en) | ![Scholar](https://img.shields.io/badge/%F0%9F%93%9A_Scholar-4285F4?style=flat-square) |
| **🏆 Kaggle** | [`Yasir Hussein Shakir`](https://www.kaggle.com/yasserhessein) | ![Competitions](https://img.shields.io/badge/%F0%9F%A5%87_Competitions-20BEFF?style=flat-square) |
| **💻 GitHub** | [`Yasir Hussein Shakir`](https://github.com/yasserhessein) | ![Code](https://img.shields.io/badge/%F0%9F%90%99_Code-181717?style=flat-square) |
| **💼 LinkedIn** | [`Yasir Hussein Shakir`](https://www.linkedin.com/in/yasir-hussein-314a65201/) | ![Professional](https://img.shields.io/badge/%F0%9F%91%94_Professional-0077B5?style=flat-square) |

</div>

---

<div align="center">

<img width="211" height="172" alt="image" src="https://github.com/user-attachments/assets/10d9690d-1fdb-4cd9-9d31-2bf2a5c3e846" />


**⭐ Star this repository if you find our research valuable!**

</div>
