## Automated Glaucoma Detection System
  -> This project aims to develop an efficient, explainable, and robust Automated Glaucoma Detection System using multiple CNN architectures, classical machine learning classifiers, and hybrid models. The models are evaluated on the RIM-ONE DL and REFUGE datasets.

## Dataset Links
  -> RIM-ONE DL Dataset: https://www.kaggle.com/datasets/shahzaddar/rim-one-dl-images

## Automated Glaucoma Detection System

This project aims to develop an efficient, explainable, and robust Automated Glaucoma Detection System using multiple CNN architectures, classical machine learning classifiers, and hybrid models. The models are evaluated on the RIM-ONE DL and REFUGE datasets.

### Batches Overview

## Batch 1: CNN Architectures (Initial Model Testing)
- **Models**: EfficientNet-B0, ResNet50, VGG16, AlexNet, InceptionV3.<br>
- **Best Performing Model**: EfficientNet-B0.<br>
- **Performance Metrics Recorded**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

**Table 1:** Performance of Various CNN Models

| Model         | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC | Confusion Matrix             |
|---------------|--------------|-----------|--------|----------|---------|-----------------------------|
| AlexNet       | 78.74        | 0.85      | 0.41   | 0.55     | 0.83    | [[114, 4], [33, 23]]        |
| VGG16         | 77.01        | 0.83      | 0.36   | 0.50     | 0.84    | [[114, 4], [36, 20]]        |
| ResNet50      | 80.46        | 0.76      | 0.57   | 0.65     | 0.82    | [[108, 10], [24, 32]]       |
| InceptionV3   | 72.41        | 0.58      | 0.50   | 0.54     | 0.71    | [[98, 20], [28, 28]]        |
| EfficientNet-B0| 78.74       | 0.77      | 0.48   | 0.59     | 0.84    | [[110, 8], [29, 27]]        |

## Batch 2: Fine-Tuning (VGG16, AlexNet)  

- **Models**: VGG16, AlexNet<br>  
- **Techniques Applied**: Adding Dropout Layers, Using Focal Loss<br>  
- **Performance Metrics Recorded**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>  

**Table 2:** Performance of VGG16 and AlexNet After Fine-Tuning  

| Model   | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC | Confusion Matrix             |
|---------|--------------|-----------|--------|----------|---------|-----------------------------|
| AlexNet | 71.84        | 0.56      | 0.55   | 0.56     | 0.68    | [[94, 24], [25, 31]]        |
| VGG16   | 71.26        | 0.56      | 0.52   | 0.54     | 0.66    | [[95, 23], [27, 29]]        |

## Batch 3: ResNet50 Fine-Tuning  

- **Models**: ResNet50<br>  
- **Techniques Applied**: Data Augmentation, Weight Adjustment<br>  
- **Performance Metrics Recorded**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>  

**Table 3:** Performance of Fine-Tuned ResNet50  

| Model   | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC | Confusion Matrix               |
|---------|--------------|-----------|--------|----------|---------|-----------------------------|
| ResNet50| 91.38        | 0.83      | 0.93   | 0.87     | 0.92    | [[107, 11], [4, 52]]       |

## Batch 4: AlexNet Random Forest Hybrid Model  

- **Models**: AlexNet + Random Forest<br>  
- **Techniques Applied**: Feature Extraction, Random Forest Classifier<br>  
- **Performance Metrics Recorded**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>  

**Table 4:** Performance of Random Forest on AlexNet Extracted Features  

| Model     | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC | Confusion Matrix             |
|-----------|--------------|-----------|--------|----------|---------|-----------------------------|
| AlexNet RF| 80.46        | 0.96      | 0.41   | 0.57     | 0.70    | [[117, 1], [33, 23]]       |

## Batch 5: EfficientNet-B0 Fine-Tuning  

- **Models**: EfficientNet-B0<br>  
- **Techniques Applied**: Using Weighted CrossEntropyLoss, Data Augmentation<br>  
- **Performance Metrics Recorded**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>  

**Table 5:** Performance of Fine-Tuned EfficientNet-B0  

| Model           | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC | Confusion Matrix       |
|-----------------|--------------|-----------|--------|----------|---------|-----------------------|
| EfficientNet-B0 | 80.00        | 1.00      | 0.53   | 0.70     | 0.77    | [[20, 0], [7, 8]]     |

## Batch 6: Cross-Validation with Random Forest & XGBoost (SMOTE Applied)

- **Models**: Random Forest (Best Hyperparameters from GridSearchCV), XGBoost (Best Hyperparameters from GridSearchCV)<br>
- **Techniques Applied**: Feature Extraction using ResNet50, SMOTE for oversampling, 5-fold Cross-Validation.<br>  
- **Performance Metrics Recorded**: Mean Accuracy, Precision, Recall, F1-Score, AUC-ROC.<br>  


- **Models**: Random Forest (Best Hyperparameters from GridSearchCV)<br>  
- **Techniques Applied**: Feature Extraction using ResNet50, SMOTE for oversampling, 5-fold Cross-Validation.<br>  
- **Performance Metrics Recorded**: Mean Accuracy, Precision, Recall, F1-Score, AUC-ROC.<br>  

**Table 6:** Cross-Validation Results for Random Forest & XGBoost with SMOTE  

| Metric         | Random Forest (Mean Score) | XGBoost (Mean Score) | LightGBM (Mean Score) |
|----------------|--------------------------|-----------------------|-----------------------|
| Accuracy (%)   | 92.54                    | 91.39                | 98.85                |
| Precision      | 0.92                     | 0.92                 | 0.98                 |
| Recall         | 0.85                     | 0.81                 | 0.98                 |
| F1-Score       | 0.88                     | 0.86                 | 0.98                 |
| AUC-ROC        | 0.91                     | 0.96                 | 0.99                 |


## Batch 1: CNN Architectures (Initial Model Testing)

  -> Models: EfficientNet-B0, ResNet50, VGG16, AlexNet, InceptionV3.<br>
  -> Best Performing Model: EfficientNet-B0.<br>
  -> Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

## Batch 2: Hybrid Model Implementation (CNN + Random Forest)
   -> Models: ResNet50 + Random Forest, EfficientNet-B0 + Random Forest.<br>
   -> Best Performing Model: ResNet50 + Random Forest.<br>
   -> Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

## Batch 4: Feature Fusion
   -> Models: ResNet50 + EfficientNet-B0 Feature Fusion.<br>
   -> Best Performing Model: ResNet50 + EfficientNet-B0 Feature Fusion.<br>
   -> Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

## Batch 5: Hyperparameter Tuning (Random Forest)
   -> Techniques: Optuna Hyperparameter Tuning.<br>
   -> Best Performing Model: Random Forest (Tuned).<br>
   -> Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

## Batch 6: Hyperparameter Tuning (XGBoost)
   -> Techniques: Optuna Hyperparameter Tuning.<br>
   -> Best Performing Model: XGBoost (Tuned).<br>
   -> Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.<br>

## Batch 7: Cross-Validation (K-Fold)
   -> Models: Random Forest, XGBoost, LightGBM.<br>
   -> Cross-Validation Technique: K-Fold (n=5).<br>
   -> Performance Metrics Recorded: Mean Accuracy, Mean Precision, Mean Recall, Mean F1-Score, Mean AUC-ROC, Average Confusion Matrix.<br>

## Batch 8: Feature Importance Analysis
   -> Models: XGBoost, LightGBM.<br>
   -> Techniques: Feature Importance Ranking.<br>
   -> Best Performing Model: XGBoost.<br>

## Top Features Identified: 10 most important features with significance scores.
  
  -> Feature 556: Importance Score = 0.0155<br>
  -> Feature 292: Importance Score = 0.0157<br>
  -> Feature 695: Importance Score = 0.0159<br>
  -> Feature 458: Importance Score = 0.0165<br>
  -> Feature 1454: Importance Score = 0.0165<br>
  -> Feature 1710: Importance Score = 0.0177<br>
  -> Feature 1902: Importance Score = 0.0232<br>
  -> Feature 1214: Importance Score = 0.0256<br>
  -> Feature 1600: Importance Score = 0.0270<br>
  -> Feature 762: Importance Score = 0.0297<br>

These feature indices correspond to specific CNN-based features extracted by the ResNet50 model. They were ranked based on their impact on model prediction performance using XGBoost's feature importance calculation.

## Possible Future Enhancements
  -Add Explainability Techniques:<br>
    -> Include Grad-CAM, SHAP, and LIME to highlight important features for glaucoma detection. This will improve interpretability and strengthen your work's clinical relevance.<br>

  -Compare Against State-of-the-Art Models:<br>
    -> Compare your best-performing models against recent state-of-the-art methods from published literature on glaucoma detection.<br>

  -Cross-Dataset Evaluation:<br>
    -> Test your models on the REFUGE dataset and compare performance to ensure robustness and generalization.<br>

  -Ablation Studies:<br>
    -> Provide ablation studies to justify your architectural choices (e.g., why EfficientNet-B0 + Random Forest works best).<br>

  -Visualizations:<br>
    -> Include t-SNE/PCA visualizations to show feature separability between glaucoma and non-glaucoma samples.<br> 
    -> Visualize confusion matrices for the best models on both datasets.<br>
 
  -Highlight Clinical Relevance:<br>
    -> Discuss the potential real-world application of your model, including deployment strategies and benefits to ophthalmologists.<br>


