# F20DL
_F20DL UG Group 2 Portfolio Repository_ \
Title: **Coffee Quality Predictor Model** \
Members: \
` Abdul Phansupkar \
Mohammed Thafheem \ 
Kasra Sartaee \
Mary Shihani \
Prudhvi Varma ` \


# Project Overview
This project explores two distinct machine learning tasks:
1. **Coffee Plant Health Detection**: Classify plant leaf images into healthy or diseased categories.
2. **Coffee Quality Prediction**: Use tabular data to predict coffee quality ratings based on sensory evaluations.

### Highlights:
- Advanced deep learning models like Convolutional Neural Networks (CNNs) and Multi-Layer Perceptrons (MLPs) are applied to image data.
- Traditional machine learning models like Logistic Regression and Decision Trees are utilized for tabular data.
- Clustering techniques (e.g., K-Means) are employed to discover patterns in coffee quality evaluations.

---
### Project Milestones
D1. Week 4 - Project Pitch \
D2. Week 11 - Project Report and code finalized \
D3. Week 12 - Project Presentation and Viva Session

## Datasets

### Image Dataset
**Coffee Plant Health Dataset**:
- Images categorized into healthy and diseased plant leaves.
- Images are resized and preprocessed with feature scaling.
- Augmentation techniques (rotation, flipping, zooming) are applied to improve model performance.
- **Source**: Dataset is to be downloaded via onedrive link: \ (https://heriotwatt-my.sharepoint.com/:u:/r/personal/pvr2000_hw_ac_uk/Documents/Datadl.zip?csf=1&web=1&e=dMfxho).
- **Original Links**: \
(https://data.mendeley.com/datasets/t2r6rszp5c/1) \
(https://data.mendeley.com/datasets/tgv3zb82nd/1) 

### Tabular Dataset
**Arabica Coffee Quality Dataset**:
- Contains sensory evaluation scores for attributes like aroma, flavor, acidity, and aftertaste.
- **Target variable**: Coffee quality (categorized into "Low", "Moderate", and "High" quality).
- **Source**: Provided in the repository as `df_arabica_clean.csv`.
- **Original Links**: 
(https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)
---

## Key Objectives

1. **Plant Health Detection**:
   - Use k-Nearest Neighbors (k-NN), Multi-Layer Perceptrons (MLPs), and Convolutional Neural Networks (CNNs) to classify images.
2. **Coffee Quality Prediction**:
   - Perform exploratory data analysis (EDA) and feature engineering.
   - Implement Logistic Regression and Decision Trees to predict coffee quality.
   - Apply K-Means clustering to discover patterns in sensory evaluations.
3. **Model Comparison**:
   - Compare the performance models using the specifc metrics: F1, recall etc;

---

## Notebook Structure

### 1. Image Processing and Classification
- **Notebook**: [cnn.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/cnn.ipynb), [KNN.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/KNN.ipynb),   [KNN Augmentation.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/KNN_Augmentation.ipynb),[MLP Notebook.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/MLP%20Notebook.ipynb)  
- Covers image preprocessing, augmentation, and training CNN, MLP, and k-NN models for plant health classification.









### 2. EDA and Coffee Quality Prediction
- **Notebook**: [EDA.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/EDA.ipynb)
, [Logistic Regression.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/Logistic%20Regression.ipynb)  , [Decision_Tree.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/Decision_Tree.ipynb)  
- Includes exploratory data analysis (EDA), clustering (K-Means), and machine learning models for coffee quality prediction.

### 3. Clustering and Insights
- **Notebook**: `[Clustering.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/Clustering.ipynb)  
- Applies K-Means clustering to identify patterns in sensory evaluation data.

---

## Getting Started

### 1. Clone the Repository
```
git clone https://github.com/abdulr2004/F20DL_UG_Group_2.git
cd F20DL_UG_Group_2
```



## 2. Dataset Preparation
Before running the models, the following notebooks must be executed:

1. **Run Preprocessing.ipynb**: [Preprocessing.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/Preprocessing.ipynb) Preprocesses the tabular dataset.
2. **Run ImagePreProcessing.ipynb**: [ImagePreProcessing.ipynb](https://github.com/abdulr2004/F20DL_UG_Group_2/blob/main/notebooks/ImagePreProcessing.ipynb) , After downlading the zip file, Prepares the image dataset.

---

## Expected Outcomes

1. A robust CNN model for detecting coffee plant health with high accuracy.
2. Machine learning models for predicting coffee quality based on sensory evaluations.
3. Insights into clustering patterns of sensory evaluations using K-Means.
4. Comparative analysis of deep learning and traditional machine learning models.

---

## Tools and Libraries Used

- **Pandas**, **NumPy**: Data handling and preprocessing.
- **Matplotlib**, **Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning models (Logistic Regression, Decision Trees, K-Means, K-NN).
- **TensorFlow/Keras**: Deep learning models (CNN, MLP).
- **Pillow**: Image processing.
