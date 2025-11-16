#  Enhancing Speaker Recognition Robustness with Scalable Deep Learning Models and MFCC Features

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-4DA51F?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.63496/ejcs.Vol1.Iss5.185-1A6FDF?style=for-the-badge&logo=doi&logoColor=white)](https://doi.org/10.63496/ejcs.Vol1.Iss5.185)

</div>

<div align="center">
  
```mermaid
graph TD
    A[Raw Audio] --> B[Preprocessing]
    B --> C[MFCC Feature Extraction]
    C --> D[Deep Learning Models]
    D --> E[FFNN 🏗️]
    D --> F[FCBP 🔄]
    D --> G[EPNN 🧠]
    E --> H[Performance Evaluation]
    F --> H
    G --> H
    H --> I[Robust Speaker Recognition]
