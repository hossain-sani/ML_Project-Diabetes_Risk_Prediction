# Type 2 Diabetes Risk Prediction with Machine Learning
*Replication of Xie et al. (Preventing Chronic Disease, CDC, 2019) using CDC BRFSS 2014 Data*

---

## 1. Paper Overview & Reference
- **Title:** Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques
- **Authors:** Zidian Xie, Olga Nikolayeva, Jiebo Luo, Dongmei Li
- **Journal:** *Preventing Chronic Disease* (CDC), September 2019; Vol 16, E130
- **DOI:** [10.5888/pcd16.190109](http://dx.doi.org/10.5888/pcd16.190109)
- **PMCID:** [PMC6795062](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6795062/)
- **Full Text (CDC):** [CDC PCD 19_0109](https://www.cdc.gov/Pcd/issues/2019/19_0109.htm)

---

## 2. Key Findings & Contributions from the Paper
1. **Machine Learning Classifiers Evaluated (8 total):**
   - Neural Network (Multilayer Perceptron)
   - Logistic Regression
   - Linear SVM
   - RBF SVM
   - Random Forest
   - Gaussian Naive Bayes
   - Polynomial SVM
   - Decision Tree
2. **Model Performance Summary:**
   - **All models achieved strong Area Under the Curve (AUC)** ranging from **0.7182 to 0.7949**.
   - **Neural Network:** Highest Accuracy (82.4%), Specificity (90.2%), and AUC (0.7949).
   - **Decision Tree:** Highest Sensitivity / Detection Rate (51.6%), preferred for initial community screening.
3. **Two Newly Identified Risk Factors (beyond traditional BMI, age, and lifestyle):**
   - **Long Sleep Duration:** Sleeping $\ge 9$ hours per day (Adjusted Odds Ratio [aOR] = 1.13, 95% CI: 1.03–1.25).
   - **Routine Checkup Frequency:** Checkup interval of less than 1 year (aOR = 2.31, 95% CI: 1.86–2.85), indicating symptomatic healthcare-seeking behavior or early detection.

---

## 3. Dataset Information
- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS), 2014
- **Raw File:** `LLCP2014.XPT` (SAS Transport format, ~645 MB uncompressed)
- **Raw Respondents:** 464,644 respondents across 279 survey variables.
- **After Preprocessing & Filtering:** Exactly **138,146 respondents**
  - **Type 2 Diabetes positive:** 20,467 (14.82%)
  - **Non-diabetic negative:** 117,679 (85.18%)
- **Target Variable:** `DIABETE3` (Binary: 1 = Type 2 Diabetes, 0 = No Diabetes)
- **Excluded:**
  - Respondents younger than 30 years (to exclude Type 1 diabetes).
  - Women diagnosed only during pregnancy (`DIABETE3 == 2`).
  - Respondents with prediabetes / borderline diabetes (`DIABETE3 == 4`).
  - Diabetic individuals diagnosed under age 30 (`DIABAGE2 < 30`).
  - Records with missing, "Don't know" or "Refused" values across the 26 selected variables.

---

## 4. The 26 Selected Risk Predictor Variables
1. `GEN_HEALTH` (`GENHLTH`): General health status (1 = Excellent to 5 = Poor).
2. `AGE_GRP` (`_AGEG5YR`): 6 condensed age categories (31–40, 41–50, 51–60, 61–70, 71–80, >80).
3. `BMI_CAT` (`_BMI5CAT`): Underweight, Normal, Overweight, Obese.
4. `CHECKUP_FREQ` (`CHECKUP1`): Routine doctor visit frequency (<1y, 1–2y, 3–5y, >5y, Never).
5. `INCOME_GRP` (`INCOME2`): 8 annual household income brackets.
6. `RACE_GRP` (`_RACE`): 8 race/ethnicity categories.
7. `METRO_STATUS` (`MSCODE`): Center city, County, Suburban, Not in MSA.
8. `FLU_SHOT` (`FLUSHOT6`): Flu vaccination in past 12 months.
9. `EMPLOYMENT` (`EMPLOY1`): Employment status.
10. `SEX` (`SEX`): Male / Female.
11. `MARITAL_STATUS` (`MARITAL`): Married, Divorced, Widowed, Separated, Never married, Unmarried couple.
12. `EDUCATION` (`_EDUCAG`): Education level.
13. `SLEEP_GRP` (`SLEPTIM1`): Hours of sleep (1–6h, 7–8h, $\ge 9$h).
14. `HEART_DISEASE` (`CVDCRHD4`): Angina or coronary heart disease history.
15. `HEALTH_COVERAGE_SRC` (`HLTHCVR1`): Primary health coverage source.
16. `MENTAL_HEALTH_GRP` (`MENTHLTH`): Days mental health not good in past 30 days (0–5 vs 6–30).
17. `KIDNEY_DISEASE` (`CHCKIDNY`): Kidney disease diagnosis.
18. `USE_EQUIP` (`USEEQUIP`): Requires special equipment (cane, wheelchair).
19. `PHYSICAL_ACTIVITY` (`_TOTINDA`): Physical activity / exercise indicator.
20. `DEPRESSION` (`ADDEPEV2`): Depressive disorder diagnosis.
21. `HOME_OWNERSHIP` (`RENTHOM1`): Own, Rent, Other.
22. `EXERCISE` (`EXERANY2`): Exercise outside work in past month.
23. `BLIND_VISION_DIFFICULTY` (`BLIND`): Serious difficulty seeing.
24. `COGNITIVE_DIFFICULTY` (`DECIDE`): Trouble concentrating, remembering, or making decisions.
25. `ANY_HEALTHCARE` (`HLTHPLN1`): Any health insurance coverage.
26. `SMOKING_STATUS` (`_SMOKER3`): 4 smoking categories (Every day, Some days, Former, Never).

---

## 5. Supervised Project Structure
```text
├── main.py                           # Manual model selection and experiment runner
├── src/                              # Downloading, preprocessing, training, prediction
├── tests/                            # Data and preprocessing checks
├── data/raw/                         # LLCP2014.XPT and source ZIP
├── data/processed/                   # Model-ready diabetes CSV
├── results/                          # CSV, ROC chart, and model_report.doc
├── docs/reference/                   # Paper and supporting reference files
└── requirements.txt                  # Python dependencies
```

## 6. Complete Execution Workflow

Run all commands from the project root:

### 6.1 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 6.2 Prepare the dataset

The raw BRFSS file is stored in `data\raw\LLCP2014.XPT`. Run:

```powershell
python src\preprocess.py
```

If the raw ZIP was split into GitHub-safe parts, recreate the XPT file with:

```powershell
python data\raw\reassemble_data.py
```

Use `--force` to recreate an existing XPT file. This joins
`LLCP2014XPT001.ZIP` and `LLCP2014XPT002.ZIP`, checks the ZIP, and extracts
`data\raw\LLCP2014.XPT`.

The preprocessing script filters invalid records, removes respondents younger
than 30, excludes pregnancy-only/prediabetes records, recodes the 26
predictors, and creates:

```text
data\processed\diabetes_brfss2014_preprocessed.csv
```

Expected preprocessing summary:

| Dataset stage | Records | Positive diabetes cases | Negative cases |
|---|---:|---:|---:|
| Raw BRFSS 2014 data | 464,644 | - | - |
| Processed model dataset | 138,146 | 20,467 | 117,679 |

To inspect the processed file manually:

```powershell
Get-Content data\processed\diabetes_brfss2014_preprocessed.csv -TotalCount 3
```

The first line displays the 26 feature columns and the target column
`DIABETES_T2`. The preprocessing script also prints the final row count,
class distribution, and number of features in the terminal.

### 6.3 Split data for training and testing

When `main.py` runs, the processed dataset is divided using a stratified
two-thirds/one-third holdout:

| Set | Records | Positive cases | Negative cases | Purpose |
|---|---:|---:|---:|---|
| Training set | 92,097 | 13,645 | 78,452 | Fit preprocessing and models |
| Test set | 46,049 | 6,822 | 39,227 | Final unbiased evaluation |

Categorical variables are one-hot encoded and numerical values are scaled.
SMOTE is applied **only to the training set**, producing 156,904 balanced
training samples. The test set remains unchanged and is never used to train
the models.

### 6.4 Select and run models

Start the manual model selector:

```powershell
python main.py
```

The program shows eleven models. Enter numbers such as `1,3,6`, or press
**Enter** to run all models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gaussian Naive Bayes
5. Linear SVM
6. Neural Network
7. LightGBM
8. CatBoost
9. XGBoost
10. Rbf SVM
11. Polynomial SVM

For direct selection without the menu:

```powershell
python main.py --models "Logistic Regression,Random Forest,Neural Network"
```

For a complete list of accepted names:

```powershell
python main.py --list-models
```

During execution, the terminal displays each model's Accuracy, Sensitivity,
Specificity, and AUC. The selected models are trained on the balanced
training data and evaluated on the untouched test data.

The three added gradient-boosting models are:

- **LightGBM:** Gradient boosting using leaf-wise tree growth.
- **CatBoost:** Gradient boosting designed to handle categorical features
  effectively; here it receives the encoded feature matrix used by the other
  models for a consistent comparison.
- **XGBoost:** Regularized gradient boosting with parallel tree construction.

### 6.5 Understand the model results

- **Accuracy:** Overall proportion of correct predictions.
- **Sensitivity:** Proportion of diabetes-positive cases detected correctly.
- **Specificity:** Proportion of non-diabetes cases identified correctly.
- **AUC:** Overall ranking ability across classification thresholds; a larger
  value generally indicates better discrimination.

For a screening project, Sensitivity is important because missing a positive
case can be costly. Specificity and AUC should also be considered when
comparing models; no single metric should be used alone.

## 7. Saved Output and Report Files

After every model run, `main.py` overwrites the result files with the models
selected in that run:

- `results/experiment_results_comparison.csv` — metrics table for selected models
- `results/roc_curves.png` — ROC curve chart
- `results/model_report.doc` — report that opens in Microsoft Word

To view the result table directly in PowerShell:

```powershell
Import-Csv results\experiment_results_comparison.csv | Format-Table
```

The CSV and report contain the model name, our Accuracy, Sensitivity,
Specificity, and AUC, alongside the corresponding values reported by Xie et
al. The report is a plain-text Word-compatible document, so it can also be
opened in a text editor when a Word processor is unavailable.

### Results table from the current experiment

| Model | Our Accuracy | Paper Accuracy | Our Sensitivity | Paper Sensitivity | Our Specificity | Paper Specificity | Our AUC | Paper AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8081 | 0.8068 | 0.5172 | 0.4634 | 0.8587 | 0.8666 | 0.8069 | 0.7932 |
| Decision Tree | 0.7390 | 0.7426 | 0.5768 | 0.5161 | 0.7673 | 0.7820 | 0.7478 | 0.7182 |
| Random Forest | 0.7919 | 0.7927 | 0.5207 | 0.5029 | 0.8390 | 0.8431 | 0.7843 | 0.7608 |
| Gaussian Naive Bayes | 0.7060 | 0.7756 | 0.6634 | 0.4876 | 0.7134 | 0.8256 | 0.7419 | 0.7598 |
| Linear SVM | 0.8082 | 0.8082 | 0.5167 | 0.4260 | 0.8589 | 0.8746 | 0.8070 | 0.7807 |
| Neural Network | 0.8155 | 0.8241 | 0.2759 | 0.3781 | 0.9094 | 0.9016 | 0.7336 | 0.7949 |
| LightGBM | 0.8172 | Not reported | 0.4925 | Not reported | 0.8737 | Not reported | 0.8077 | Not reported |
| CatBoost | 0.8173 | Not reported | 0.4919 | Not reported | 0.8739 | Not reported | 0.8046 | Not reported |
| XGBoost | 0.8105 | Not reported | 0.5085 | Not reported | 0.8631 | Not reported | 0.8019 | Not reported |
| Rbf SVM | 0.8166 | 0.8178 | 0.3530 | 0.4014 | 0.8972 | 0.8902 | 0.7603 | 0.7788 |
| Polynomial SVM | 0.7962 | 0.7962 | 0.4974 | 0.4515 | 0.8481 | 0.8561 | 0.7861 | 0.7587 |

## 8. Risk Prediction Demo

```powershell
python src\predict.py
```

This is an educational demonstration and is not a medical diagnosis.
