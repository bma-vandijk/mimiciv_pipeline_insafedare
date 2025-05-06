import pandas as pd
import os
import numpy as np


def get_cohort(
    icd_code: str = "I50",
    root: bool = True,
    demographic_data: bool = True,
    admission_data: bool = True,
    diagnoses_data: bool = True,
    omr_data: bool = True,
    medication_data: bool = True,
    data_path: str = "mimiciv/2.0",
    icd_mapping_path: str = "utils",
):
    """
    Selects cohort of hospital visits where a patient received the input ICD code.
    Adds additional data from tables, e.g., patients, admissions, diagnoses, omr, medication.
    """
    PATHS = {
        "mapper": icd_mapping_path,
        "patients": os.path.join(data_path, "hosp", "patients.csv.gz"),
        "admissions": os.path.join(data_path, "hosp", "admissions.csv.gz"),
        "diagnoses": os.path.join(data_path, "hosp", "diagnoses_icd.csv.gz"),
        "omr": os.path.join(data_path, "hosp", "omr.csv.gz"),
        "prescriptions": os.path.join(data_path, "hosp", "prescriptions.csv.gz"),
    }

    # create list of icd9 and icd10 codes on which to select patients
    mapping = pd.read_csv(PATHS["mapper"], header=0, delimiter="\t")
    if root:
        mapping[["diagnosis_code", "icd9cm", "icd10cm"]] = mapping[
            ["diagnosis_code", "icd9cm", "icd10cm"]
        ].apply(lambda col: col.str[:3])
    mapping = mapping[mapping.icd10cm.str.startswith(icd_code)]
    disease_codes = mapping.icd9cm.to_list() + mapping.icd10cm.to_list()
    disease_codes = list(set(disease_codes))

    # load diagnoses table and select relevant patients
    print("selecting cohort...")
    diagnoses = pd.read_csv(PATHS["diagnoses"], compression="gzip", header=0)
    if root:
        diagnoses.icd_code = diagnoses.icd_code.str[:3]

    visits = diagnoses[diagnoses.icd_code.isin(disease_codes)]
    visits = visits[["subject_id", "hadm_id"]]
    visits = visits.groupby(["subject_id", "hadm_id"]).first().reset_index(drop=False)

    # load and merge patient info and health markers according to input
    if demographic_data:
        print("adding demographic data...")
        info = _load_demographics(PATHS["patients"])
        visits = visits.merge(info, how="left", on=["subject_id"])
        # filter for only adults
        visits = visits.loc[visits["age"] >= 18]

    if admission_data:
        print("adding admissions data...")
        info = _load_admissions(PATHS["admissions"])
        visits = visits.merge(info, how="left", on=["subject_id", "hadm_id"])

    if diagnoses_data:
        print("adding #diagnoses data...")
        info = _load_n_diagnoses(PATHS["diagnoses"])
        visits = visits.merge(info, how="left", on=["hadm_id"])

    if omr_data:
        print("adding omr data...")
        info = _load_omr(PATHS["omr"])
        for omr in info:
            visits = visits.sort_values("admittime")
            omr = omr.sort_values("chartdate")
            visits = pd.merge_asof(
                visits,
                omr,
                by="subject_id",
                left_on="admittime",
                right_on="chartdate",
                direction="nearest",
            )

    if medication_data:
        print("adding #medication data...")
        info = _load_medications(PATHS["prescriptions"])
        visits = visits.merge(info, how="left", on=["hadm_id"])

    return visits


def _load_n_diagnoses(path: str):
    # get number of diagnoses for each hospital visit
    diagnoses = pd.read_csv(path, compression="gzip", header=0)
    diagnoses = diagnoses.groupby(["hadm_id"]).count()[["seq_num"]]
    diagnoses = diagnoses.rename({"seq_num": "n_diagnoses"}, axis=1)
    diagnoses = diagnoses.reset_index(drop=False)
    return diagnoses


def _load_admissions(path: str):
    admissions = pd.read_csv(
        path,
        compression="gzip",
        header=0,
        index_col=None,
        parse_dates=["admittime", "dischtime"],
    )
    admissions["los"] = admissions["dischtime"] - admissions["admittime"]
    admissions["los_hours"] = admissions["los"].dt.total_seconds() / 3600
    # admissions["los_hours"] = admissions["los_hours"].round(0)
    return admissions


def _load_demographics(path: str):
    demographics = pd.read_csv(
        path,
        compression="gzip",
        header=0,
        index_col=None,
        usecols=[
            "subject_id",
            "anchor_year",
            "anchor_age",
            "anchor_year_group",
            "dod",
            "gender",
        ],
    )
    demographics["yob"] = demographics["anchor_year"] - demographics["anchor_age"]
    demographics["min_valid_year"] = demographics["anchor_year"] + (
        2019 - demographics["anchor_year_group"].str.slice(start=-4).astype(int)
    )

    demographics = demographics.rename({"anchor_age": "age"}, axis=1)

    return demographics


def _load_omr(path: str):

    omr = pd.read_csv(path)
    omr.chartdate = pd.to_datetime(omr.chartdate)

    # get BMI
    bmi = omr[omr["result_name"].str.startswith("BMI")]
    bmi = bmi.rename({"result_value": "bmi"}, axis=1)
    bmi = bmi[["subject_id", "chartdate", "bmi"]]
    bmi.bmi = bmi.bmi.astype(float)

    # get bp
    bp = omr[omr["result_name"].str.startswith("Blood Pressure")]
    bp = bp.rename({"result_value": "bp"}, axis=1)
    bp = bp[["subject_id", "chartdate", "bp"]]
    systolic_diastolic = bp.bp.str.split("/", expand=True)
    systolic, diastolic = systolic_diastolic.iloc[:, 0], systolic_diastolic.iloc[:, 1]
    systolic.name = "bp_systolic"
    diastolic.name = "bp_diastolic"
    bp = pd.concat([bp, systolic, diastolic], axis=1)
    bp = bp.drop("bp", axis=1)
    bp.bp_systolic, bp.bp_diastolic = bp.bp_systolic.astype(
        float
    ), bp.bp_diastolic.astype(float)

    return bmi, bp


def _load_medications(path: str):
    # load prescriptions table
    medications = pd.read_csv(
        path,
        compression="gzip",
        header=0,
        usecols=["subject_id", "hadm_id"],
        dtype=np.int32,
    )
    # get # medications per admission
    medications = (
        medications.groupby("hadm_id").count()["subject_id"].reset_index(drop=False)
    )
    medications = medications.rename({"subject_id": "n_medications"}, axis=1)

    return medications
