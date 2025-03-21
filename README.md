🔬 Automated glaucoma screening using CNNs, Vision Transformers, and ensemble classifiers on ACRIMA & ORIGA datasets.

📌 Overview
GlaucoFusion is an AI-powered system that detects glaucoma from retinal fundus images using a hybrid deep learning pipeline. It extracts deep features using CNNs (VGG16/ResNet) and optionally Vision Transformers, and classifies using an ensemble of Random Forest, XGBoost, and LightGBM. The goal is to assist in early diagnosis and clinical decision-making for glaucoma, one of the leading causes of irreversible blindness.

🗂️ Datasets Used
🧾 ACRIMA Dataset (~700 fundus images)

Binary classification: Glaucoma / Non-glaucoma
High-resolution fundus images
Source: ACRIMA on GitHub
🧾 ORIGA Dataset (~650 fundus images)

Includes disc/cup annotations for CDR computation
Binary glaucoma classification labels
Source: ORIGA Dataset Info
💡 Datasets were preprocessed, normalized, and merged to create a diverse training set.

🛠️ Features
🔍 Deep feature extraction with VGG16 / ResNet / Vision Transformers
🌲 Classification using:
Random Forest
XGBoost
LightGBM
🧠 Ensemble Strategies:
Soft Voting
Stacking (Logistic Regression as meta-learner)
📊 Model evaluation with AUC-ROC, Confusion Matrix, Precision, Recall
🧪 Easy to plug-in Grad-CAM or SHAP for explainability
⚙️ Ready for real-world clinical deployment and telemedicine integration
