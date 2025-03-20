# GlaucoNet-Multi-Modal-Deep-Learning-for-Explainable-Glaucoma-Detection
Glaucoma is a progressive eye disease that leads to irreversible vision loss if not detected early. GlaucoNet is an AI-powered, explainable glaucoma detection system that combines deep learning with ensemble learning to improve accuracy and interpretability.

📂 PAPILA Dataset Structure Breakdown
The PAPILA_DBv1 dataset contains clinical data, fundus images, and segmentation masks for 244 patients (both eyes). It is structured as follows:

1️⃣ ClinicalData/

Contains spreadsheet files with clinical parameters for 244 patients.
Each patient has two records (one for the right eye (OD) and one for the left eye (OS )
Data includes:
  Age, Gender
  Diagnosis (0 = Healthy, 1 = Glaucoma, 2 = Suspicious)
  Refractive error
  Phakic/Pseudophakic status (Lens removed or not)
  Intraocular pressure (IOP)
  Pachymetry (Corneal thickness)
  Axial length
  Mean defect (Visual field loss measurement)

2️⃣ ExpertsSegmentations/

Contains manual OD (optic disc) and OC (optic cup) segmentations.
Two expert ophthalmologists provided annotations.
Each segmentation file is stored in plain text format (X, Y coordinates of contours).
Naming format:
scss
Copy
Edit
RETXXX_OD_cup_exp1.txt   # Patient XXX, Right Eye, Cup, Annotated by Expert 1
RETXXX_OS_disc_exp2.txt  # Patient XXX, Left Eye, Disc, Annotated by Expert 2

3️⃣ FundusImages/

Contains retinal fundus images for both left and right eyes.
488 images in JPEG format (each patient has both eyes captured).
Image size: 2576 × 1934 pixels
Naming format follows the segmentation files (e.g., RETXXX_OD.jpg for the right eye).

4️⃣ HelpCode/

Contains Python scripts and notebooks to help researchers process the dataset.
Likely includes utilities for:
Reading and visualizing fundus images
Loading and overlaying segmentation masks
Extracting clinical features for analysis

💡 How to Use PAPILA in Your AI Model

✅ Multi-Modal Learning:

Combine fundus image features with clinical data (CDR, IOP, MD, etc.).
Use CNN (ResNet, EfficientNet, Vision Transformers) for feature extraction from images.
Use Random Forest or another ML model to process numerical clinical data.

✅ Explainability with XAI:

Apply Grad-CAM on fundus images.
Use SHAP to analyze feature importance from clinical data.
Implement Graph Attention Networks (GATs) to model relationships between features.

✅ Segmentation-Assisted Diagnosis:

Use optic disc and optic cup segmentation to calculate VCDR, HCDR, and neuroretinal rim area.
Train a segmentation model to predict OD/OC boundaries for CDR computation.
