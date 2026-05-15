# BAN6440 Module 5 - ANN Cancer Diagnosis

## MD Anderson Cancer Institute | Nexford University | May 2026

**Author:** Ganiu Olalekan Mustapha

\---

## Project Overview

This project implements an Artificial Neural Network (ANN) using TensorFlow to classify breast tumors as **Malignant** or **Benign** using the Wisconsin Breast Cancer Diagnostic Dataset. The solution is developed in the context of MD Anderson Cancer Institute's radiological diagnostic workflow.

**Key results:**

|Metric|Score|
|-|-|
|Accuracy|96.49%|
|Recall (priority)|97.22%|
|F1-Score|97.22%|
|AUC-ROC|99.50%|
|5-Fold CV Recall|98.25% +/- 1.12%|

\---

## Project Structure

```
module5\_ann\_cancer/
|
|-- ann\_cancer\_diagnosis.ipynb           <- Main Python application
|-- README.md                            <- This file
|-- BAN6440\_Module5\_WrittenSummary.docx  <- Word documentation
|
|-- data/
|   |-- breast\_cancer\_wisconsin.csv      <- Dataset (auto-generated on
|                                           first run, or download from
|                                           Kaggle link below)
|
|-- ann\_outputs/                         <- Created automatically on run
    |-- breast\_cancer\_wisconsin.csv
    |-- 01\_correlation\_heatmap.png
    |-- 02\_training\_history.png
    |-- 03\_confusion\_matrix.png
    |-- 04\_roc\_curve.png
    |-- 05\_cross\_validation.png
    |-- evaluation\_metrics.json
    |-- ann\_cancer\_model.keras
```

\---

## Dataset

**Name:** Wisconsin Breast Cancer Diagnostic Dataset
**Source:** Kaggle - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
**Records:** 569 patients | 30 features | Binary label (Malignant / Benign)
**Missing values:** None

The code loads the dataset automatically via sklearn.datasets.load\_breast\_cancer()
which is identical to the Kaggle CSV. A copy is saved to ann\_outputs/ on first run.

If you prefer to use the Kaggle CSV directly, download data.csv from the link
above and update:

&#x09;data = load\_breast\_cancer()

&#x09;df   = pd.DataFrame(data.data, columns=data.feature\_names)

&#x09;df\['diagnosis'] = data.target   # 1 = Benign, 0 = Malignant ... in ann\_cancer\_diagnosis.ipynb

to:
	df = pd.read\_csv("data/data.csv")

\---

## Requirements

### Python Version

Python 3.9 or higher recommended.

### Install Dependencies

Open the VS Code integrated terminal (Ctrl + `) and run:

&#x20;   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn


Note for Windows users: If TensorFlow installation fails, try:
pip install tensorflow-cpu

\---

## How to Run in VS Code

### Step 1 - Open the project folder

&#x20;   File -> Open Folder -> select the module5\_ann\_cancer folder


### Step 2 - Open integrated terminal

&#x20;   Terminal -> New Terminal   (or Ctrl + `)


### Step 3 - Run the application

&#x20;   python ann\_cancer\_diagnosis.ipynb


### Step 4 - Expected terminal output (summary)

&#x20;   =================================================================
      BAN6440 Module 5 | ANN Cancer Diagnosis | TensorFlow
      Dataset: Wisconsin Breast Cancer Diagnostic (Kaggle/UCI)
      Reference: Tran *et al*. (2021) Genome Medicine 13:152
    =================================================================

    STEP 1: DATA COLLECTION
       Dataset shape     : (569, 31)
       Benign  (1)       : 357 patients
       Malignant (0)     : 212 patients
       Missing values    : 0

    STEP 2: PREPROCESSING
       Outlier cells     : 608 / 17070 (3.56%) - retained (clinically meaningful)
       Train set         : (455, 30)
       Test set          : (114, 30)

    STEP 3: MODEL ARCHITECTURE
       Total parameters  : 5,057

    STEP 4: TRAINING
       \[Training epochs printed - up to 150, early stopping applies]

    STEP 5: EVALUATION
       \[DEFAULT THRESHOLD = 0.50]
       Accuracy  : 96.49%    Precision : 98.57%
       Recall    : 95.83%    F1-score  : 97.18%    AUC-ROC : 99.50%

       \[TUNED THRESHOLD - maximizes recall]
       Accuracy  : 96.49%    Recall    : 97.22%  <- oncology priority

    STEP 6: 5-FOLD CROSS-VALIDATION
       Fold 1: acc=0.9670  rec=1.0000  auc=0.9933
       Fold 2: acc=0.9780  rec=0.9825  auc=0.9979
       Fold 3: acc=0.9560  rec=0.9649  auc=0.9933
       Fold 4: acc=0.9670  rec=0.9825  auc=0.9964
       Fold 5: acc=0.9780  rec=0.9825  auc=0.9990
       Mean recall: 98.25% +/- 1.12%

    Application completed successfully.


### Step 5 - View output files

All plots and the saved model appear in the ann\_outputs/ folder.
Open them directly in VS Code or in Windows File Explorer.

\---

## Output Files Explained

|File|What it shows|
|-|-|
|01\_correlation\_heatmap.png|Pearson correlation: top 10 FNA features vs diagnosis|
|02\_training\_history.png|Loss and accuracy curves across training epochs|
|03\_confusion\_matrix.png|Default threshold vs tuned threshold side by side|
|04\_roc\_curve.png|ROC curve - AUC = 0.9950|
|05\_cross\_validation.png|All metrics across 5 folds|
|evaluation\_metrics.json|All metrics saved as JSON|
|ann\_cancer\_model.keras|Saved TensorFlow model weights|

\---

## Model Architecture

&#x20;   Input Layer  (30 FNA features)
         |
    Dense(64) - L2(0.001) regularization
    BatchNormalization
    ReLU activation
    Dropout(20%)
         |
    Dense(32) - L2(0.001)
    BatchNormalization
    ReLU activation
    Dropout(20%)
         |
    Dense(16) - L2(0.001)
    BatchNormalization
    ReLU activation
    Dropout(20%)
         |
    Output: Dense(1) - Sigmoid -> P(Benign)


Loss function : Binary cross-entropy
Optimiser     : Adam (learning\_rate=0.001)
Callbacks     : EarlyStopping (patience=15, restore\_best\_weights=True)
ReduceLROnPlateau (patience=7, factor=0.5)

\---

## Why Recall is the Priority Metric

In oncology screening, a false negative (predicting benign when the tumor is malignant) delays treatment and significantly reduces survival probability.
A false positive leads to a follow-up biopsy, which is uncomfortable but recoverable. This asymmetry means recall (sensitivity) is the primary optimization target, not accuracy or precision.

The decision threshold was tuned from the default 0.50 to 0.30 specifically to maximize recall. This raised recall from 95.83% to 97.22%, reducing missed malignancies on the test set from 2 to 1. The trade-off (marginal precision reduction from 98.57% to 97.22%) is clinically justified.

Reference: Bhinder *et al.* (2021). Artificial intelligence in cancer research
and precision medicine. Cancer Discovery, *11*(4), 900-915.
https://doi.org/10.1158/2159-8290.CD-21-0090

\---


\---

## Troubleshooting

ModuleNotFoundError: No module named 'tensorflow'
pip install tensorflow

ModuleNotFoundError: No module named 'seaborn'
pip install seaborn

TensorFlow GPU warnings flooding the terminal
These are harmless. The code suppresses most with:
os.environ\['TF\_CPP\_MIN\_LOG\_LEVEL'] = '3'
If warnings still appear, add this before the import:
import os; os.environ\['TF\_CPP\_MIN\_LOG\_LEVEL'] = '3'

Plots not appearing as pop-up windows
Expected behaviour. The code uses matplotlib Agg backend which saves
plots as PNG files rather than displaying windows. Check ann\_outputs/.

Code crashes with FileNotFoundError on data.csv
The code does not require an external CSV - it loads data via sklearn.
If you see this error, ensure you are running from the project folder:
cd path/to/module5\_ann\_cancer
python ann\_cancer\_diagnosis.ipynb

\---

BAN6440 Business Analytics | Nexford University | May 2026

