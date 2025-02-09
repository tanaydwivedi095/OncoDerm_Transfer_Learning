# OncoDerm: Transfer Learning Phase

This phase of the OncoDerm project focuses on **Transfer Learning** for skin cancer classification. We leverage pre-trained Convolutional Neural Networks (CNNs) to classify skin lesions as Malignant or Benign, enhancing diagnostic accuracy with deep learning.

## Overview
- Utilizes CNN architectures such as **VGG16, VGG19, EfficientNetB3, EfficientNetV2B3, DenseNet121, ResNet50, and Xception**.
- **CNNs are used for feature extraction**, capturing key patterns from dermoscopic images.
- Extracted features are then **fed into a Random Forest Classifier (RFC)** to perform the final classification.
- Designed to run efficiently on **systems with low computational resources**.
- Evaluates and compares the performance of different models.

## Dataset Details
The dataset consists of dermoscopic images labeled as either **Malignant** or **Benign**. Each image undergoes preprocessing, including resizing, normalization, and augmentation to enhance model robustness.

## Models Implemented
1. **VGG16**
2. **VGG19**
3. **EfficientNetB3**
4. **EfficientNetV2B3**
5. **DenseNet121**
6. **ResNet50**
7. **Xception**

## Model Workflow
1. **Feature Extraction**: CNN architectures extract relevant features from input images.
2. **Classification**: Extracted features are passed through a **Random Forest Classifier (RFC)** for final classification.
3. **Performance Evaluation**: The combined CNN+RFC pipeline is assessed using standard evaluation metrics.

## Performance Results
Each model is evaluated using metrics such as:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

Final performance comparisons help determine the best-performing model for the next phase.

## Next Phase: Transformer-Based Models
The Transformer phase of this project explores **vision transformers (ViTs)** for skin cancer classification. Check it out here:
🔗 [OncoDerm: Transformer Phase](https://github.com/tanaydwivedi095/OncoDerm_Transformers)

## Installation & Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/tanaydwivedi095/OncoDerm_TransferLearning.git
cd OncoDerm_TransferLearning
```

### Step 2: Install Dependencies
Since `requirements.txt` is not available, manually install the necessary packages:
```bash
pip install tensorflow torch torchvision torchaudio scikit-learn pandas numpy matplotlib streamlit
```

### Step 3: Run the Jupyter Notebook
Execute the transfer learning models using:
```bash
jupyter notebook
```
Open and run `DenseNet121+RFC.ipynb` or other model-specific notebooks.

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

