

# 🧬 ProGIGene: Multi-Omics Risk Stratification Pipeline

**ProGIGene** is a machine learning pipeline for **progression risk modeling in early-stage gastrointestinal (GI) cancers**, integrating multi-omics features including transcriptomics, DNA methylation, copy number variations, somatic mutations, and proteomics. It implements a transparent and interpretable two-stage modeling strategy using survival analysis and ElasticNet classification.

---

## 🚀 Features

- Preprocessing of multi-omics data with variance filtering and standardization  
- Cox proportional hazards survival modeling (time-to-event analysis)  
- Binary risk label creation from continuous survival risk scores  
- ElasticNet-regularized logistic regression for feature selection and classification  
- Automated top-250 feature extraction based on coefficient magnitude  
- Performance evaluation with precision–recall and permutation testing  
- Threshold tuning to balance precision and recall  
- SHAP-ready feature outputs for model explainability

---

## 🛠️ Tech Stack

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- LightGBM
- imbalanced-learn (SMOTE)
- scikit-survival
- SHAP

---

## 📁 Dataset

Place a file named `final_training_dataset.csv` in the root directory. The dataset must include:
- `sample`: unique sample ID  
- `early_progression_label`: binary progression label  
- `pfi`: progression-free interval event indicator  
- `pfi_time`: progression-free survival time (in days)  
- Molecular features across omics platforms

---
```
📁 ProGIGene/
├── 📁 data_processing/ 
│ ├── 💊 clinical_data.py
│ ├── 🧬 copy_number_variation.py
│ ├── 📝 final_training_dataset.py
│ ├── 🧬 gene_expression.py
│ ├── 🧫 immune_subtype.py
│ ├── 🧬 methylation.py
│ ├── 🧬 mutation.py
│ ├── 🔗 pathways.py
│ └── 📊 protein_expression.py
│
├── 📁 external_cohort_validation/ 
│ └── 🌐✅ geo_cohort_validation.py
│
├── 📁 model
│ ├── 🧠 gene_only_train.py
│ ├── 🧠 multi_omics_train.py
│ └── 📈 plots.py

```
---
### `data_processing/clinical_data.py`
- Data Link: [Xena](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp)
- Loads and cleans raw clinical data by renaming columns, selecting key features, and filtering for early-stage gastrointestinal cancers with valid progression-free interval times.
- Generates a binary label indicating early disease progression within one year and exports the processed dataset for further analysis - `early_stage_clinical_labeled_dataset.csv`

### `data_processing/copy_number_variation.py`
- Loads and preprocesses [CNV](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz) and clinical data(`early_stage_clinical_labeled_dataset.csv`), applies thresholding to convert continuous CNV values into discrete categories (-1, 0, 1), and filters features with variation in at least 10 samples.

- Merges CNV data with early progression labels, applies variance filtering and scaling, then trains a regularized logistic regression model (ElasticNet) to identify predictive features.

- Selects the top 300 CNV features based on model coefficients and exports the resulting dataset for downstream analysis or modeling `copy_number_variation.csv`

### `data_processing/gene_expression.py`
- Processes [gene expression](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz) and clinical data(`early_stage_clinical_labeled_dataset.csv`) by applying log transformation, selecting highly variable genes, and identifying co-expression modules correlated with early cancer progression.

- Trains an ElasticNet-regularized logistic regression model to select the top 500 predictive genes, which are saved `na_exp.csv` for downstream analysis or model development.

### `data_processing/immune_subtype.py`

- Loads and standardizes [immune subtype](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Subtype_Immune_Model_Based.txt.gz) labels by formatting them into lowercase, underscore-separated categories with a consistent 'immune_' prefix.

- Filters the immune subtype data to include only clinical samples(`early_stage_clinical_labeled_dataset.csv`) of interest and saves the cleaned dataset(`immune_subtype.csv`) for downstream analysis.

### `data_processing/methylation.py`

-   Loads, filters, and merges methylation data across multiple cancer types - [ESCA](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.ESCA.sampleMap%2FHumanMethylation450.gz), [COAD](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.COAD.sampleMap%2FHumanMethylation450.gz), [LIHC](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LIHC.sampleMap%2FHumanMethylation450.gz), [PAAD](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.PAAD.sampleMap%2FHumanMethylation450.gz), [READ](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.READ.sampleMap%2FHumanMethylation450.gz), [STAD](https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.STAD.sampleMap%2FHumanMethylation450.gz), selecting the top 2000 high-variance CpG sites and aligning them with clinical progression labels(`early_stage_clinical_labeled_dataset.csv`).

-   Preprocesses data by imputing missing values and scaling, then trains an elastic net logistic regression model to identify and export the top 500 predictive CpG features(`methylation_top_shap.csv`).

### `data_processing/mutation.py`

- Filters out silent mutations from [mutation data set](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz) and clinical data (`early_stage_clinical_labeled_dataset.csv`), converts gene mutation data into a binary matrix per sample, and retains only genes mutated in at least 5 samples.

- Merges with clinical labels, adds mutation counts per sample, and exports the processed dataset(`gene_mutation.csv`) for downstream analysis.

### `data_processing/pathways.py`

- Cleans and standardizes pathway names, transposes the [gene-pathway matrix](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/merge_merged_reals_sample_level.txt.gz), and formats column names with a pathway_ prefix for consistency mapped with clinical data (`early_stage_clinical_labeled_dataset.csv`)

- Filters the data to include only clinical samples of interest and exports the processed pathway features(`early_stage_gene_pathway.csv`) for further analysis.


### `data_processing/protein_expression.py`

-   Loads and transposes the [protein expression matrix](https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA-RPPA-pancan-clean.xena.gz), prefixes features, filters for clinical samples(`early_stage_clinical_labeled_dataset.csv`), and fills missing values.

-   Selects the top 300 most variable proteins across samples and exports the processed dataset(`protein_expression.csv`) for downstream modeling.

### `data_processing/final_training_dataset.py`

-   Merges clinical(`early_stage_clinical_labeled_dataset.csv`), immune subtype(`immune_subtype.csv`), and multi-omics datasets (RNA(`rna_exp.csv`), mutation(`gene_mutation.csv`), CNV(`copy_number_variation.csv`), methylation(`methylation_top_shap.csv`), protein(`protein_expression`), and pathway(`early_stage_gene_pathway.csv`)) into a unified feature matrix.

-   Applies one-hot encoding to categorical clinical variables, fills missing values with zeros, and exports the final dataset(`final_training_dataset.csv`) for model training.


### `model/gene_only_train.py`

-   Trains an ElasticNet logistic regression on RNA data with variance filtering and scaling, selects top 250 features, and evaluates via precision-recall metrics.

-   Assesses robustness by permutation testing, comparing true AUC to shuffled-label AUCs to calculate empirical p-values.

### `model/multi_omics_train.py`

-   _Survival-informed Classification_: Builds a binary classifier using ElasticNet logistic regression on Cox survival risk scores to predict high vs. low risk patients after filtering and scaling.

-   _Model Evaluation & Validation_: Validates the model with precision-recall curves and permutation tests to confirm robustness and significance.

### `model/plots.py`

-   _ROC Curve_: Plots model performance by showing true vs. false positive rates and AUC.

-   _PCA_: Reduces top variable features to 3 components, visualizing them colored by survival correlation.

### `external_cohort_validation/geo_cohort_validation.py`
-   Downloads GEO datasets, maps probes to genes, extracts RNA features, and computes risk scores with a trained logistic regression model.

- Processes survival data, assigns risk groups, and evaluates prognosis using Kaplan–Meier curves and AUC metrics in validation cohorts.

## ⚙️ Pipeline Overview

```text
1. Load and sanitize feature names
2. Split data into train/test (stratified)
3. Apply variance threshold filtering
4. Standardize features with z-score scaling
5. Train CoxPH survival model → generate risk scores
6. Convert risk scores into binary labels (top 33% high-risk)
7. Use ElasticNet logistic regression to select top features
8. Re-train classifier on top features
9. Tune threshold for best F1 score
10. Validate robustness via permutation testing
```

---

## 📈 Example Metrics

- **C-index**: ~0.82  
- **AUC (ROC)**: ~0.81  
- **Precision/Recall tuning**: enables flexible sensitivity/specificity balancing  
- **Permutation testing**: verifies statistical significance beyond chance

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Or import the code in a notebook for step-by-step execution and visualization.

---

## 🧠 Example Snippet

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
cox_model = CoxPHSurvivalAnalysis(alpha=0.01)
cox_model.fit(X_train_scaled, y_train_surv)
risk_scores = cox_model.predict(X_test_scaled)
```

---

## 📊 Output

- Predicted probabilities for early progression
- Risk stratification thresholds
- Precision-recall curves
- SHAP-ready feature importance values (optional)

