# 🔬 Glaucoma Detection using Multi-Modal AI

Early Diagnosis with CDR, Deep Features & Ensemble Learning
🧠 ORIGA & ACRIMA Datasets | 🔍 Explainability with Grad-CAM (Ongoing)

✅ Project Overview
This research project focuses on building a multi-modal AI pipeline for glaucoma detection by combining:

1) Clinical features (like Cup-to-Disc Ratio - CDR)

2) Line two Deep image features extracted using ResNet50

3) Line three Ensemble learning, threshold tuning, and explainability tools

# 📌 Datasets Used

🟢 ORIGA Dataset: Fundus images + Segmentation masks + Labels

🔵 ACRIMA Dataset: Fundus images + Glaucoma labels (No masks)

# 🎯 Completed Modules So Far

1️⃣ Data Preprocessing

✔ Extracted Cup Area, Disc Area, and computed CDR from ORIGA segmentation masks

✔ Cleaned noisy/missing masks by skipping unusable files

✔ Normalized CDR and saved metadata in cdr_values_fixed.csv

2️⃣ Clinical Feature Integration

✔ Merged CDR values with ORIGA metadata into origa_final.csv

✔ Normalized CDR → CDR_Norm for ML compatibility

3️⃣ Classical Machine Learning (CDR-Only)

✔ Trained Random Forest and XGBoost on CDR-only features

✔ Tuned models and evaluated with:

Accuracy

F1-score (focused on Glaucoma class)

✔ Applied SMOTE to handle class imbalance and improve recall

4️⃣ Deep Feature Extraction

✔ Extracted ResNet50 features from ORIGA fundus images

✔ Saved image embeddings in resnet_features.csv

5️⃣ Feature Fusion

✔ Combined CDR + Deep Features → merged_features.csv

6️⃣ Advanced Modeling

✔ Trained on fused features using:

    1) Random Forest (Tuned)

    2) XGBoost (Tuned with F1-score as the target)

    3) Stacking Ensemble (RF + XGB + Logistic Regression)

    4) Soft Voting Ensemble

✔ Performance:

✅ Stacked Model Accuracy: ~76%

✅ F1-score (Glaucoma class): ~0.59

7️⃣ Threshold Tuning
✔ Performed custom threshold search to optimize Glaucoma detection
✔ Achieved:

High recall for Glaucoma class

Balanced precision & F1-score using best threshold ≈ 0.17–0.29

8️⃣ K-Fold Cross Validation
✔ Applied K-Fold CV during hyperparameter tuning to improve generalization & reduce variance

9️⃣ Model Training from Scratch
✔ Trained ResNet50 on ORIGA images from scratch for upcoming Grad-CAM visualizations
✔ Built a custom PyTorch Dataset class with error handling for missing images

