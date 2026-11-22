# Personal Loan Classification and Customer Segmentation
**Mahmoud Abusahyoun | Artificial Intelligence coursework, HTU**

Predicts bank personal-loan acceptance using KNN and a Keras neural network, then explores customer segments with K-Means and PCA. This portfolio edition uses the more complete original `Mahmoud_Abusahyoun_AI_code.ipynb`, including classification reports and customer segmentation.

## Data and methods
`data/bank_personal_loan_data.csv` is the supplied 5,000-row dataset. The later standalone upload was byte-identical to the ZIP copy. `ID` and `ZIP Code` are removed. Exact duplicate records are removed before a stratified split; no rows are synthesized. Upstream dataset ownership/license details were not supplied and are not asserted here.

- Reserve 20% for retrospective testing and use 80% for development.
- Select KNN's `k` from 1–20 using stratified five-fold CV on training data. Scaling is fitted within each fold, using the original five continuous columns.
- Split the development data again for ANN training/validation (64%/16% of the deduplicated full dataset). Fit its scaler only on the ANN fitting subset. Keep the original 16/8-unit architecture and 20 epochs.
- Explore clusters on the complete feature dataset, excluding the loan target from fitting. Select cluster count using silhouette scores; use loan acceptance only for post-fit interpretation.

Clustering statistics describe the observed dataset and are not held-out predictive metrics. The original classification work already inspected test results; the corrected split and procedure do not substitute for independent external validation. Accuracy should be read alongside positive-class precision, recall and F1 because loan acceptance is imbalanced.

## Corrected results
| Model | Test accuracy | Positive-class F1 |
|---|---:|---:|
| KNN | 96.59% | 0.7952 |
| Neural network | 96.99% | 0.8438 |

These values come from the executed notebook and `results/test_comparison.csv`, not the original saved outputs.

## Run and review
Use Python 3.12 and install `requirements.txt`, then open `personal_loan_ai.ipynb` with this folder as the working directory. The executed notebook already contains results and charts. `artifacts/` includes the KNN pipeline, Keras model, ANN preprocessor, and fitted clustering model/scaler. `results/` includes class reports, cluster profiles, loan rates and execution details.

Seeds are fixed for NumPy, TensorFlow, splitters and K-Means. Package versions used in this CPU run are pinned; different hardware can still change numerical results.

## Corrections
Scaling now occurs after splitting and within KNN CV. KNN tuning no longer reads test labels. ANN validation is explicit and has training-only preprocessing. The final clustering count now follows the measured silhouette result instead of conflicting with it. Old metrics were replaced by rerun results.
