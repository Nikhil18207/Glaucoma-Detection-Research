Batch 1: CNN Architectures (Initial Model Testing)
Models: EfficientNet-B0, ResNet50, VGG16, AlexNet, InceptionV3.

Best Performing Model: EfficientNet-B0.

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Batch 2: ResNet50 Feature Extraction + XGBoost
Extracted features using ResNet50 from your retinal fundus images.

Applied XGBoost Classifier on these extracted features.

Hyperparameter Tuning using GridSearchCV.

Evaluated using 5-Fold Cross-Validation.

Performance Metrics Recorded: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix.

Visualizations: ROC Curve (not plotted yet).

Batch 3: ResNet50 Feature Extraction + LightGBM
Extracted features using ResNet50 (same as above).

Applied LightGBM Classifier on these extracted features.

Hyperparameter Tuning using GridSearchCV.

Evaluated using Single Run (No Cross-Validation).

Achieved Highest Accuracy of 98.85% with AUC-ROC of 0.9868.

Visualizations: ROC Curve (not plotted yet).

Batch 4: Comparative Analysis of All Models
Compared Initial CNN Models (EfficientNet, ResNet50, VGG16, AlexNet, InceptionV3) against XGBoost and LightGBM.

Structured tables summarizing performance for each model.

Provided insight into the best performing models for each stage.

Batch 5: AUC-ROC Curve Plotting (In Progress)
Need to generate AUC-ROC Curves for:

EfficientNet-B0, ResNet50, VGG16, AlexNet, InceptionV3.

XGBoost (Cross-Validation averaged AUC).

LightGBM (Single run AUC).

