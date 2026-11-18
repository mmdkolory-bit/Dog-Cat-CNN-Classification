# Dog-Cat-CNN-Classification
Binary image classification (Dog vs. Cat) using a custom CNN, including evaluation metrics and layer ablation study.
# Dog and Cat Image Classification using CNN

This project focuses on binary image classification (Dog vs. Cat) using a Convolutional Neural Network (CNN). The model is trained and evaluated on a dataset provided by Kaggle.

## 📊 Dataset
The dataset used in this project can be found here:
[Dog and Cat Classification Dataset - Kaggle](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

## 🎯 Project Objectives
- Design and train a custom Convolutional Neural Network (CNN).
- Evaluate the model's performance using the test dataset.
- Plot the Mean Squared Error (MSE) loss curve against the number of epochs.
- Calculate and report key classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- **Ablation Study:** Investigate the effect of different CNN layers (e.g., adding/removing Conv2D or Pooling layers) on the overall performance of the model.

## 📁 Project Structure
- `House_Prices_CNN_Project.ipynb` (or your main notebook): Contains the code for data preprocessing, model architecture, training, and evaluation.
- `train.py`: Python script for training the model.
- `requirements.txt`: List of required Python libraries.
- `REPORT_TEMPLATE_FA.md`: Project report and findings.

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
