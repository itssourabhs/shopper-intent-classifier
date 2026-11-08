# Shopper Intent Lab: predicting purchases from online shopping sessions

Machine Learning, Assignment 2 · M.Tech AI & ML · BITS Pilani WILP · First Semester 2026-27

| | |
|---|---|
| **Live Streamlit app** | https://shopper-intent-classifier-ml-bits-assignment-2.streamlit.app |
| **GitHub repository** | https://github.com/itssourabhs/shopper-intent-classifier |

---

## a. Problem statement

An online store records every browsing session: which kinds of pages the visitor opened, how long
they stayed, how often visitors leave from those pages, the month, the visitor's browser and region,
and whether the session ended in a purchase.

**Task: binary classification.** Given the 17 attributes of a session, predict whether it ends in a
purchase (`Revenue = 1`) or not (`Revenue = 0`).

The difficulty is **class imbalance**: only 15.6% of sessions buy. A model that answers
"no purchase" every time is 84.4% accurate and completely useless. So this project compares five
classic classifiers on the same data using six metrics, and treats **MCC** (Matthews Correlation
Coefficient, which uses all four cells of the confusion matrix) as the deciding metric rather than
accuracy. The trained models are deployed in an interactive Streamlit app that scores any uploaded
test file.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository, id 468 — <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset> |
| Citation | Sakar, C.O., Polat, S.O., Katircioglu, M. & Kastro, Y. (2019). *Neural Computing and Applications*, 31, 6893–6908 |
| Instances | **12,330** sessions (12,205 after removing 125 exact duplicate rows) |
| Features | **17** (10 numeric, 7 categorical / binary) |
| Target | `Revenue` — 1,908 purchase sessions (15.5%) vs 10,422 non-purchase (84.5%) |
| Missing values | None |
| Collection | Each session belongs to a different user over one year, so the data is not dominated by one campaign, day or user |

**Features**

| Feature | Type | Meaning |
|---|---|---|
| `Administrative`, `Informational`, `ProductRelated` | numeric (count) | Number of pages of each type visited in the session |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | numeric (seconds) | Time spent on each page type |
| `BounceRates` | numeric (0–0.2) | Average share of visitors who entered on these pages and left without any other request |
| `ExitRates` | numeric (0–0.2) | Average share of page views on these pages that were the last in a session |
| `PageValues` | numeric | Average value of the pages visited before a completed transaction (Google Analytics) |
| `SpecialDay` | numeric (0–1) | Closeness of the visit to a special day such as Mother's Day or Valentine's Day |
| `Month` | categorical | Month of the visit (10 values; January and April are absent) |
| `OperatingSystems`, `Browser`, `Region`, `TrafficType` | categorical (integer IDs) | Visitor's OS, browser, region, and the traffic source that brought them |
| `VisitorType` | categorical | `Returning_Visitor`, `New_Visitor` or `Other` |
| `Weekend` | binary | Whether the visit was on a weekend |

**Preprocessing and split**

1. **Drop 125 exact duplicate rows** (all non-buyers), so a copy cannot sit in train while its twin sits in test.
2. **Stratified 80 / 20 split** → 9,764 training rows and **2,441 test rows**, both 15.6% buyers. `random_state = 5984`.
3. Inside each model's pipeline:
   - numeric features → `log1p` then `StandardScaler`. The counts and durations are extremely right-skewed (`ProductRelated_Duration` reaches 63,974 s); without the log, a few marathon sessions dominate KNN distances and the SVM kernel.
   - `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, `TrafficType` → one-hot encoding. The four integer columns are IDs, not quantities, so they are not treated as numbers. Categories never seen in training are ignored.
   - `Weekend` → 0 / 1.
   - The result is **73 model inputs**.
4. `test_data.csv` is exactly the 2,441-row hold-out set: raw columns + `Revenue`. No model saw it during training or tuning.

**Hyperparameter tuning:** 5-fold stratified cross-validated grid search on the training rows only, choosing by MCC.

| Model | Chosen hyperparameters | CV MCC |
|---|---|---|
| Logistic Regression | `C = 0.01`, `class_weight = balanced` | 0.6045 |
| Decision Tree | `max_depth = 5`, `min_samples_leaf = 30`, `class_weight = balanced` | 0.5930 |
| KNN | `n_neighbors = 25`, `weights = uniform` | 0.5612 |
| Naive Bayes (Gaussian) | `var_smoothing = 0.1` | 0.4477 |
| SVM (RBF kernel) | `C = 0.3`, `class_weight = balanced`, Platt-scaled probabilities | 0.6004 |

## c. GitHub Repository Link

https://github.com/itssourabhs/shopper-intent-classifier

```
shopper-intent-classifier/
├── app.py                    Streamlit app: upload CSV → choose model → metrics, confusion matrix, ROC
├── train.py                  loads data, tunes and trains all 5 models, writes model/ and test_data.csv
├── bits_lab_run.ipynb        the training run as executed on BITS Virtual Lab
├── requirements.txt          pinned to the versions the models were saved with
├── test_data.csv             2,441-row hold-out test set (features + Revenue)
├── data/
│   └── online_shoppers_intention.csv   original UCI file
├── model/
│   ├── logistic_regression.joblib      each file = preprocessing + classifier pipeline
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   ├── test_metrics.csv                the comparison table below
│   └── training_summary.json           chosen hyperparameters, CV scores, versions
├── shopintent/               shared code, used by both train.py and app.py
│   ├── schema.py             column groups and input cleaning
│   ├── models.py             preprocessing pipeline, the 5 classifiers, search grids
│   └── evaluation.py         the six metrics
└── tests/test_shopintent.py  metric hand-checks, saved-model reproducibility, headless app run
```

## d. Models used

All five models were trained on the same 9,764 rows and evaluated on the same 2,441-row hold-out set
(`test_data.csv`), with a 0.50 probability threshold.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8685 | 0.9169 | 0.5550 | 0.8063 | **0.6574** | **0.5949** |
| Decision Tree | 0.8451 | **0.9207** | 0.5031 | 0.8429 | 0.6301 | 0.5687 |
| KNN | **0.8947** | 0.9115 | **0.7163** | 0.5419 | 0.6170 | 0.5646 |
| Naive Bayes | 0.7522 | 0.8729 | 0.3751 | **0.8770** | 0.5255 | 0.4571 |
| Support Vector Machine | 0.8898 | 0.9187 | 0.6409 | 0.6728 | 0.6564 | 0.5911 |

Bold = best in column. For reference, always predicting "no purchase" scores accuracy 0.8435 and MCC 0.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | **Best overall**: highest MCC (0.5949) and F1 (0.6574), and also the best cross-validated MCC (0.6045), so the result is not a lucky split. CV chose strong regularisation (`C = 0.01`) with balanced class weights. The balancing pushes recall to 0.806 (it catches 308 of the 382 buyers) at the cost of 247 false alarms (precision 0.555). The largest coefficient is `PageValues` (+1.39), followed by `Month = Nov` (+0.80): a visitor who has viewed pages that usually come before a sale, during the November shopping season, is the most likely buyer. A linear boundary is enough because one feature carries most of the signal. |
| **Decision Tree** | **Highest AUC (0.9207) but only third on MCC (0.5687).** The tuned tree (depth 5, at least 30 rows per leaf, balanced weights) splits first on `PageValues`, which accounts for 81% of its total importance. Sessions with `PageValues > 0` buy 56% of the time, against 3.9% when it is 0. The tree ranks sessions well, but its 30 leaves can produce at most 30 distinct probabilities, so the 0.5 cut is coarse: recall 0.843 with precision 0.503, roughly one false alarm for every real buyer. Deeper and unpruned trees scored lower in cross-validation because they overfit the training rows. |
| **KNN** | **Highest accuracy (0.8947) and precision (0.7163), but the lowest recall (0.5419): it misses 46% of buyers.** With `k = 25` and no class weighting, non-buyers outnumber buyers 5.4 : 1, so most neighbourhoods vote "no purchase". Distances are also diluted across 73 dimensions, 62 of them one-hot columns. Its top accuracy mostly reflects the 84% majority class, and the gap between its accuracy rank (1st) and its MCC rank (4th) is the clearest example in this table of why accuracy misleads on imbalanced data. It is also the largest model to ship (553 KB), because it stores all 9,764 training rows. |
| **Naive Bayes** | **Weakest on every metric except recall, where it is best (0.8770).** Accuracy 0.7522 is *below* the 0.8435 of always predicting "no purchase", and precision 0.375 means about 5 of every 8 sessions it flags are false alarms. The cause is the "naive" assumption that features are independent given the class, which this data badly violates: `BounceRates` and `ExitRates` correlate at 0.91, and `ProductRelated` and its duration at 0.86. The same evidence is counted twice and the probabilities become overconfident. CV picked the largest smoothing on offer (`var_smoothing = 0.1`) to soften this. It is fast and simple but poorly suited to correlated features. |
| **Support Vector Machine** | **A close second**: F1 0.6564 and MCC 0.5911, within 0.004 of Logistic Regression, with the most even precision / recall balance (0.641 / 0.673) and the second-best accuracy (0.8898). It uses an RBF kernel, `C = 0.3` and balanced weights. An SVM outputs distances, not probabilities, so it was wrapped in Platt scaling (`CalibratedClassifierCV`) to compute AUC; at a 0.5 cut the calibrated probabilities favour precision more than Logistic Regression does. The non-linear kernel gains nothing over the linear model on this data, and it is the most expensive: 14 s to tune versus 2.5 s or less for the others, and about 0.5 s to score the test set versus under 50 ms. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It has the best MCC and F1 on the hold-out set and the best cross-validated MCC, and it is also the simplest, fastest and most explainable model, since its coefficients say why a session was flagged. The SVM is statistically level but slower and opaque. If the business goal were instead "never miss a buyer" (for example, to trigger a discount pop-up), the higher recall of the Decision Tree or Naive Bayes would be worth trading precision for, or equivalently Logistic Regression's threshold could be lowered. The app's threshold slider shows this trade-off live. |

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py          # ~45 s: re-tunes, retrains and overwrites model/ and test_data.csv
streamlit run app.py     # opens the app at http://localhost:8501
pip install pytest && pytest -q    # 15 tests
```

**Streamlit app features:** CSV upload (with column validation, plus a bundled `test_data.csv` and
a download button for it) · model selection dropdown · the six metrics as tiles · confusion matrix ·
ROC curve with the current operating point · classification report · side-by-side comparison of all
five models on the uploaded file · decision-threshold slider. The app only loads the saved
`.joblib` pipelines and predicts; it never retrains on uploaded data.

**Deployment:** Streamlit Community Cloud → New app → this repository, branch `main`, file `app.py`,
Python 3.13 (Advanced settings). `requirements.txt` pins `scikit-learn==1.9.1`, the version that
saved the models.
