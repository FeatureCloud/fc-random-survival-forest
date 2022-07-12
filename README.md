# Random Survival Forest

## Description
This FeatureCloud app implements a Random Survival Forest based on the scikit-survival library.  

It was originally developed by Sarah Remus

## Input
CSV file with samples as rows, features as columns, including a column containing the duration information and event information.

## Output
- C-index evaluation
- Feature importance

## Workflows
This is a standalone app. It is not compatible with other apps.

## Config
```json
fc_rsf:
  files:
    input: "train_2_2_even.csv"
    input_test: "test_2_2_even.csv"
  parameters:
    time_column: "time"
    event_column: cens
    n_estimators_local: 1000
    min_sample_leafes: 15
    min_sample_split: 10
    iterations_fi: 2
    min_concordant_pairs: 20
    random_state: "random"
    merge_test_train: true
```

## Privacy
- No patient-level data is exchanged (more infos: https://pubmed.ncbi.nlm.nih.gov/26159465/)
- Exchanges:
  - Local Random Forest models
  - Local c-indices  
- No additional privacy-enhancing techniques implemented yet