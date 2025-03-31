## Automated Glaucoma Detection System

This project aims to develop an efficient, explainable, and robust Automated Glaucoma Detection System using multiple CNN architectures, classical machine learning classifiers, and hybrid models. The models are evaluated on the RIM-ONE DL and REFUGE datasets.

Dataset Links

RIM-ONE DL Dataset: Link

REFUGE Dataset: Link

Batches Overview

Batch 1: CNN Architectures (Initial Model Testing)

Models: EfficientNet-B0, ResNet50, VGG16, AlexNet, InceptionV3.

Best Performing Model: EfficientNet-B0.

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 2: Hybrid Model Implementation (CNN + Random Forest)

Models: ResNet50 + Random Forest, EfficientNet-B0 + Random Forest.

Best Performing Model: ResNet50 + Random Forest.

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 3: Ensemble Learning

Models: Soft Voting, Hard Voting, Stacking Ensemble.

Best Performing Model: Stacking Ensemble (ResNet50 + Random Forest + EfficientNet-B0).

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 4: Feature Fusion

Models: ResNet50 + EfficientNet-B0 Feature Fusion.

Best Performing Model: ResNet50 + EfficientNet-B0 Feature Fusion.

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 5: Hyperparameter Tuning (Random Forest)

Techniques: Optuna Hyperparameter Tuning.

Best Performing Model: Random Forest (Tuned).

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 6: Hyperparameter Tuning (XGBoost)

Techniques: Optuna Hyperparameter Tuning.

Best Performing Model: XGBoost (Tuned).

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 7: Cross-Validation (K-Fold)

Models: Random Forest, XGBoost, LightGBM.

Cross-Validation Technique: K-Fold (n=5).

Performance Metrics Recorded: Mean Accuracy, Mean Precision, Mean Recall, Mean F1-Score, Mean AUC-ROC, Average Confusion Matrix.

Batch 8: Feature Importance Analysis

Models: XGBoost, LightGBM.

Techniques: Feature Importance Ranking.

Best Performing Model: XGBoost.

Top Features Identified: 10 most important features with significance scores.
