# Plan for v2

## Objective
Fix the metadata issue from v1 so the notebook can execute successfully on Kaggle GPUs.

## Action Items
1.  Copy `ecg_pipeline_v1.ipynb` to `v2`.
2.  Inject the required Jupyter Notebook `kernelspec` and `language_info` metadata via a JSON modification script.
3.  Update the Kaggle metadata to use `v2`.
4.  Push to Kaggle and monitor the training execution of the 4 architectures.
