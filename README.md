# Simple pipeline for preprocessing MIMIC-IV

This (demo) pipeline uses MIMIC-IV version 1.0 and creates from the admissions, patients (core) and diagnosis_icd 'hosp' tables,
a cohort based on a specific disease (heart failure by default) that were readmitted after first visit (within 30 day 'gap' by default). 

Re-purposed and simplified version of code from [Gupta et al. (2022)](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/tree/main)

The main output from sandbox.ipynb is currently case_df, which concerns readmissions of all adult patients with heart failure, control_df with patients that were not readmitted or readmitted beyond the 30 day gap.   

## Use

1. Create Python >= 3.12 venv
2. Install dependencies with requirements.txt
3. Sandbox .ipynb implements the necessary data preprocessors in select_disease_cohort.py and select_patient_info.py 