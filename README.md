# Simple pipeline for preprocessing MIMIC-IV

This simple pipeline creates a disease cohort from the [MIMIC-IV version 2.0](https://physionet.org/content/mimiciv/2.0/) database from a ICD10 code, and adds various demographic and medical features (e.g. blood pressure, BMI, ethnicity, length of stay, age) from the *admissions*, *patients*, and *diagnosis_icd* tables with which various ML tasks can be done. 

# Use

1. Create Python >= 3.12 venv
2. Install dependencies from requirements.txt
3. Check and run mortality_prediction.ipynb if you want to create and *interact* with the resulting dataset, and ML model
4. Check and run pipeline_sandbox.ipynb and pipeline.py for the pipeline components that were made to align with the INSAFEDARE pipeline requirements, but do essentially the same things as the files mentioned in the previous step. 

# Dataset
MIMIC-IV is open and free to use once you made an account on phsionet.org and followed some modules at the [CITI Program](https://physionet.org/about/citi-course/). If you follow these steps carefully the required modules can be done for free and MIMIC-IV can be downloaded and used. It should however not be published somewhere online where others can download it or see its data (e.g. stdout). So the dataset is just for internal use.

We follow the structure of the dataset in the mimiciv folder:
"mimiciv/2.0/hosp/diagnoses_icd.csv.gz"

For now only admissions.csv.gz, patients.csv.gz, omr.csv.gz, prescriptions.csv.gz, and diagnoses_icd.csv.gz have to be downloaded to create the dataset -- downloads are slow and other tables are not currently used.

# Under construction