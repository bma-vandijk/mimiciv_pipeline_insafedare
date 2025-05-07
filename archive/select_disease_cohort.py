"""Module for selecting and processing disease cohorts from MIMIC-IV data.

This module provides functions to:
- Load and process admission data
- Convert ICD-9 codes to ICD-10
- Select patient cohorts based on specific disease codes
"""

import os
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

# Constants
GROUP_COL = "subject_id"  # Identifier for patients
VISIT_COL = "hadm_id"  # Identifier for visits
ADMIT_COL = "admittime"  # Admission time
DISCH_COL = "dischtime"  # Discharge time
DISEASE_LABEL = "N18"  # or I50 = Heart Failure, # I25 = Coronary Artery Disease, # N18 = Chronic Kidney Disease, # J44 = Chronic obstructive pulmonary disease
VERSION = "2.0"  # MIMIC-IV version 1.0 or 2.0

# Paths to MIMIC-IV data files
PATH_ADMISSIONS: str = os.path.join("mimiciv", VERSION, "core", "admissions.csv.gz")
PATH_DIAGNOSES_ICD: str = os.path.join(
    "mimiciv", VERSION, "hosp", "diagnoses_icd.csv.gz"
)
PATH_PATIENTS: str = os.path.join("mimiciv", VERSION, "core", "patients.csv.gz")
PATH_ICD_MAP: str = os.path.join("utils", "ICD9_to_ICD10_mapping.txt")


def get_admissions_df(
    path_admissions_df: str = PATH_ADMISSIONS,
    admit_col: str = "admittime",
    disch_col: str = "dischtime",
    filter_deaths: bool = True,
) -> pd.DataFrame:
    """Load and process admissions data from MIMIC-IV.

    Args:
        path_admissions_df: Path to the admissions CSV file
        admit_col: Column name for admission time
        disch_col: Column name for discharge time

    Returns:
        DataFrame containing processed admissions data with:
        - Length of stay in hours
        - Only non-expired admissions
    """
    visit = pd.read_csv(
        path_admissions_df,
        compression="gzip",
        header=0,
        index_col=None,
        parse_dates=[admit_col, disch_col],
    )

    # Calculate length of stay
    visit["los"] = visit[disch_col] - visit[admit_col]
    visit["los_hours"] = visit["los"].dt.total_seconds() / 3600
    visit["los_hours"] = visit["los_hours"].apply(
        lambda x: int(x + 0.5)
    )  # Round up if more than 0.5

    # Remove hospitalizations with death
    if filter_deaths:
        visit = visit.loc[visit.hospital_expire_flag == 0]

    return visit


def read_icd_mapping_df(map_path: str = PATH_ICD_MAP) -> pd.DataFrame:
    """Read and process ICD-9 to ICD-10 mapping table.

    Args:
        map_path: Path to the mapping file

    Returns:
        DataFrame containing ICD-9 to ICD-10 mappings with lowercase descriptions
    """
    mapping = pd.read_csv(map_path, header=0, delimiter="\t")
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def get_diagnosis_icd_df(module_path: str = PATH_DIAGNOSES_ICD) -> pd.DataFrame:
    """Load diagnosis ICD codes from MIMIC-IV.

    Args:
        module_path: Path to the diagnoses ICD file

    Returns:
        DataFrame containing diagnosis ICD codes
    """
    return pd.read_csv(module_path, compression="gzip", header=0)


def standardize_icd(
    mapping: pd.DataFrame,
    diag: pd.DataFrame,
    map_code_col: str = "diagnosis_code",
    root: bool = True,
) -> None:
    """Convert ICD-9 codes to ICD-10 in the diagnosis DataFrame.

    Args:
        mapping: DataFrame containing ICD-9 to ICD-10 mappings
        diag: DataFrame containing diagnosis of patients (diagnoses they were billed for)
        map_code_col: Column name containing the codes in mapping DataFrame
        root: If True, only use first 3 digits of ICD-9 code for mapping

    Modifies:
        diag DataFrame in place, adding:
        - root_icd10_convert: Converted ICD-10 codes
        - root: First 3 digits of converted ICD-10 codes
    """
    count = 0
    code_cols = mapping.columns
    errors: List[str] = []

    def icd_9to10(icd: str) -> Optional[str]:
        """Convert single ICD-9 code to ICD-10.

        Args:
            icd: ICD-9 code to convert

        Returns:
            Converted ICD-10 code or None if not found
        """
        if root:
            icd = icd[:3]

        if map_code_col not in code_cols:
            errors.append(f"ICD NOT FOUND: {icd}")
            return None

        matches = mapping.loc[mapping[map_code_col] == icd]
        if matches.shape[0] == 0:
            errors.append(f"ICD NOT FOUND: {icd}")
            return None

        return mapping.loc[mapping[map_code_col] == icd].icd10cm.iloc[0]

    # Create new column with original codes as default
    col_name = "root_icd10_convert"
    diag[col_name] = diag["icd_code"].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    for code, group in diag.loc[diag.icd_version == 9].groupby(by="icd_code"):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            diag.at[idx, col_name] = new_code

        count += group.shape[0]
        print(f"{count}/{diag.shape[0]} rows processed")

    # Add column for just the roots of the converted ICD10 codes
    diag["root"] = diag[col_name].apply(lambda x: x[:3] if isinstance(x, str) else None)


def standardize_codes_and_select_cohort(
    disease_label: str = "I50",
    path_admissions_df: str = PATH_ADMISSIONS,
    filter_deaths: bool = True,
) -> pd.DataFrame:
    """Select patient cohort based on specific disease code.

    Args:
        disease_label: ICD-10 code to filter by (e.g., 'I50' for heart failure)
        path_admissions_df: Path to admissions data file

    Returns:
        DataFrame containing admissions for patients with the specified disease
    """
    diag_df = get_diagnosis_icd_df()
    mapping_df = read_icd_mapping_df()
    visit = get_admissions_df(path_admissions_df, filter_deaths=filter_deaths)

    standardize_icd(mapping_df, diag_df, root=True)

    # Select patients with at least one record of the given ICD-10 code
    diag_df.dropna(subset=["root"], inplace=True)
    pos_ids = pd.DataFrame(
        diag_df.loc[diag_df.root.str.contains(disease_label)].hadm_id.unique(),
        columns=["hadm_id"],
    )

    visit = visit[visit["hadm_id"].isin(pos_ids["hadm_id"])]
    print(f"[ READMISSION DUE TO {disease_label} ]")

    return visit
