# BAN6440 Module 5 — ANN Cancer Diagnosis

MD Anderson Cancer Institute | Nexford University | May 2026  
Author: Ganiu Olalekan Mustapha

\---

# Project Overview

This project implements an Artificial Neural Network (ANN) using TensorFlow to classify breast tumours as Malignant or Benign using the Wisconsin Breast Cancer Diagnostic Dataset. The solution is developed in the context of MD Anderson Cancer Institute's radiological diagnostic workflow.

## Key results (from actual code output)

|Metric|Score|
|-|-|
|Accuracy|95.61%|
|Recall (priority)|95.83%|
|F1-Score|96.50%|
|AUC-ROC|99.21%|
|5-Fold CV Recall|97.90% ± 1.31%|

\---

# Project Structure

```text
module5_ann_cancer/
|
|-- ann_cancer_diagnosis.py              <- Main Python application
|-- ann_cancer_diagnosis.ipynb           <- Notebook version of the python application
|-- README.md                            <- This file
|-- BAN6440_Module5_WrittenSummary.docx  <- Word documentation
|
|-- data/
|   |-- breast_cancer_wisconsin.csv      <- Dataset (auto-generated on
|                                           first run, or download from
|                                           Kaggle link below)
|
|-- ann_outputs/                         <- Created automatically on run
    |-- breast_cancer_wisconsin.csv
    |-- 01_correlation_heatmap.png
    |-- 02_training_history.png
    |-- 03_confusion_matrix.png
    |-- 04_roc_curve.png
    |-- 05_cross_validation.png
    |-- evaluation_metrics.json
    |-- ann_cancer_model.keras
```

\---

# Dataset

**Name:** Wisconsin Breast Cancer Diagnostic Dataset  
**Source:** Kaggle — https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
**Records:** 569 patients | 30 features | Binary label (Malignant / Benign)  
**Missing values:** None

The code loads the dataset automatically via `sklearn.datasets.load\_breast\_cancer()` which is identical to the Kaggle CSV. A copy is saved to `ann\_outputs/` on first run.

If you prefer to use the Kaggle CSV directly, download `data.csv` from the link above and replace the data loading block in `ann\_cancer\_diagnosis.py` with:

```python
df = pd.read\_csv("data/data.csv")
```

\---

# Requirements

## Python Version

Python 3.9 or higher recommended.

## Install Dependencies

Open the VS Code integrated terminal (`Ctrl + `` `) and run:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### Note for Windows users:

If TensorFlow installation fails, try:

```bash
pip install tensorflow-cpu
```

\---

# How to Run in VS Code

## Step 1 — Open the project folder

`File -> Open Folder -> select the module5\_ann\_cancer folder`

## Step 2 — Open integrated terminal

`Terminal -> New Terminal   (or Ctrl + \\`)`

## Step 3 — Run the application

```bash
python ann\_cancer\_diagnosis.py
```

## Step 4 — Expected terminal output (summary)

```text
=================================================================
  BAN6440 Module 5 | ANN Cancer Diagnosis | TensorFlow
  Dataset: Wisconsin Breast Cancer Diagnostic (Kaggle/UCI)
  Reference: Bhinder et al. (2021) Cancer Discovery 11(4)
=================================================================

STEP 1: DATA COLLECTION
   Dataset shape     : (569, 31)
   Benign  (1)       : 357 patients
   Malignant (0)     : 212 patients
   Missing values    : 0

STEP 2: PREPROCESSING
   Outlier cells     : 608 / 17070 (3.56%) — retained (clinically meaningful)
   Train set         : (455, 30)
   Test set          : (114, 30)

STEP 3: MODEL ARCHITECTURE
   Total parameters  : 5,057

STEP 4: TRAINING
   \[Training epochs printed — up to 150]
   Training stopped at epoch: 130
   Best weights restored from epoch: 115
   LR reductions at epochs: 51, 122, 129

STEP 5: EVALUATION
   \[DEFAULT THRESHOLD = 0.50]
   Accuracy  : 95.61%    Precision : 97.18%
   Recall    : 95.83%    F1-score  : 96.50%    AUC-ROC : 99.21%

   \[TUNED THRESHOLD = 0.30]
   Accuracy  : 95.61%    Recall    : 95.83%  <- oncology priority
   Note: tuned and default produced identical metrics on this test set.
   See Section 6 of written summary for clinical interpretation.

STEP 6: 5-FOLD CROSS-VALIDATION
   Fold 1: acc=0.9670  prec=0.9655  rec=0.9825  f1=0.9739  auc=0.9917
   Fold 2: acc=0.9780  prec=0.9825  rec=0.9825  f1=0.9825  auc=0.9985
   Fold 3: acc=0.9670  prec=0.9821  rec=0.9649  f1=0.9735  auc=0.9923
   Fold 4: acc=0.9670  prec=0.9500  rec=1.0000  f1=0.9744  auc=0.9969
   Fold 5: acc=0.9780  prec=1.0000  rec=0.9649  f1=0.9821  auc=0.9974
   Mean recall: 97.90% +/- 1.31%

  Application completed successfully.
```

## Step 5 — Screenshot the terminal

Take a screenshot of the final metrics block. This is your evaluation evidence for submission.

## Step 6 — View output files

All plots and the saved model appear in the `ann\_outputs/` folder.

\---

# Output Files Explained

|File|What it shows|
|-|-|
|01\_correlation\_heatmap.png|Pearson correlation: top 10 FNA features vs diagnosis|
|02\_training\_history.png|Loss and accuracy curves across training epochs|
|03\_confusion\_matrix.png|Default threshold vs tuned threshold side by side|
|04\_roc\_curve.png|ROC curve — AUC = 0.9921|
|05\_cross\_validation.png|All metrics across 5 folds|
|evaluation\_metrics.json|All metrics saved as JSON|
|ann\_cancer\_model.keras|Saved TensorFlow model weights|

\---

# Model Architecture

```text
Input Layer  (30 FNA features)
     |
Dense(64) — L2(0.001) regularisation
BatchNormalization
ReLU activation
Dropout(20%)
     |
Dense(32) — L2(0.001)
BatchNormalization
ReLU activation
Dropout(20%)
     |
Dense(16) — L2(0.001)
BatchNormalization
ReLU activation
Dropout(20%)
     |
Output: Dense(1) — Sigmoid -> P(Benign)
```

* Loss function: Binary cross-entropy
* Optimiser: Adam (learning\_rate=0.001)
* Callbacks: EarlyStopping (patience=15, restore\_best\_weights=True)
* ReduceLROnPlateau (patience=7, factor=0.5)

\---

# Why Recall is the Priority Metric

In oncology screening, a false negative — predicting benign when the tumour is malignant — delays treatment and significantly reduces survival probability. A false positive leads to a follow-up biopsy, which is uncomfortable but recoverable. This asymmetry means recall (sensitivity) is the primary optimisation target, not accuracy or precision.

A threshold sweep from 0.30 to 0.70 was conducted to maximise recall. On this test set, the tuned threshold (0.30) produced metrics identical to the default (0.50), with recall remaining at 95.83%. This finding is clinically significant in itself: it indicates the model's probability calibration is already well-aligned with oncology screening priorities at the default threshold, requiring no aggressive relaxation to achieve high sensitivity.

Reference: Bhinder, B., Gilvary, C., Madhukar, N. S., \& Elemento, O. (2021). Artificial intelligence in cancer research and precision medicine. *Cancer Discovery, 11*(4), 900–915. https://doi.org/10.1158/2159-8290.CD-21-0090



\---

# Troubleshooting

## ModuleNotFoundError: No module named 'tensorflow'

```bash
pip install tensorflow
```

## ModuleNotFoundError: No module named 'seaborn'

```bash
pip install seaborn
```

## TensorFlow GPU warnings flooding the terminal

These are harmless. The code suppresses most with:

```python
os.environ\['TF\_CPP\_MIN\_LOG\_LEVEL'] = '3'
```

## Plots not appearing as pop-up windows

Expected behaviour. The code uses matplotlib Agg backend which saves plots as PNG files. Check `ann\_outputs/`.

## Code crashes with FileNotFoundError on data.csv

The code does not require an external CSV — it loads data via sklearn. Ensure you are running from the project folder:

```bash
cd path/to/module5\_ann\_cancer
python ann\_cancer\_diagnosis.py
```

\---

BAN6440 Business Analytics | Nexford University | May 2026

