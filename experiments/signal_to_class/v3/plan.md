# Plan for v3

## Objective
Fix the dataset path issue from v2. The Kaggle environment threw a "Data files not found" error because the hardcoded dataset paths might be slightly different in the dynamic mount.

## Action Items
1.  Implement `glob` searching to dynamically locate `mitbih_train.csv` and `mitbih_test.csv` anywhere inside `/kaggle/input/`.
2.  Print the discovered paths to the console for debugging.
3.  Push the updated `ecg_pipeline_v3.ipynb` to Kaggle and monitor execution.
