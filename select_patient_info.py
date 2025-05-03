"""Module for processing and analyzing patient data from MIMIC-IV database.

This module provides functionality to:
- Process patient demographic and visit information
- Identify and label readmission cases based on specified time gaps
- Get comorbidities for each visit
"""

import os
from typing import Tuple, Optional, List
import pandas as pd
import datetime
from tqdm import tqdm

# Paths to MIMIC-IV data files
VERSION: str = '2.0'
PATH_ADMISSIONS: str = os.path.join("mimiciv", VERSION, "core", "admissions.csv.gz")
PATH_DIAGNOSES_ICD: str = os.path.join("mimiciv", VERSION, "hosp", "diagnoses_icd.csv.gz")
PATH_PATIENTS: str = os.path.join("mimiciv", VERSION, "core", "patients.csv.gz")
PATH_ICD_MAP: str = os.path.join("utils", "ICD9_to_ICD10_mapping.txt")
PATH_OMR: str = os.path.join("mimiciv", VERSION, "hosp", "omr.csv.gz")
PATH_PRESCRIPTIONS: str = os.path.join("mimiciv", VERSION, "hosp", "prescriptions.csv.gz")

# Pre-loading some dfs for row-based feature operations
omr = pd.read_csv(PATH_OMR)
pres_df = pd.read_csv(PATH_PRESCRIPTIONS, compression="gzip", header=0)
# First, filter OMR to only include BMI measurements, convert dates to datetime
bmi_omr = omr[omr['result_name'] == 'BMI (kg/m2)']
bmi_omr['chartdate'] = pd.to_datetime(bmi_omr['chartdate'])

# Blood pressure
# First, filter OMR to only include blood pressure measurements, convert dates to datetime
bp_omr = omr[omr['result_name'] == 'Blood Pressure']
bp_omr['chartdate'] = pd.to_datetime(bp_omr['chartdate'])


def get_patient_df(
    path_patients_df: str = PATH_PATIENTS,
    visit_df: Optional[pd.DataFrame] = None,
    group_col: str = 'subject_id',
    visit_col: str = 'hadm_id',
    admit_col: str = 'admittime',
    disch_col: str = 'dischtime',
    version: str = '1.0',
) -> pd.DataFrame:
    """Process and merge patient demographic data with visit information.
    
    Args:
        path_patients_df: Path to the patients CSV file
        visit_df: DataFrame containing visit information
        group_col: Column name for patient identifier
        visit_col: Column name for visit identifier
        admit_col: Column name for admission time
        disch_col: Column name for discharge time
        
    Returns:
        DataFrame containing merged patient and visit information for adult patients
    """
    if visit_df is None:
        raise ValueError("visit_df cannot be None")
        
    # Read patient demographic data
    pts = pd.read_csv(
        path_patients_df,
        compression='gzip',
        header=0,
        index_col=None,
        usecols=['subject_id', 'anchor_year', 'anchor_age', 'anchor_year_group', 'dod', 'gender']
    )

    # Calculate year of birth and minimum valid year
    pts['yob'] = pts['anchor_year'] - pts['anchor_age']
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))

    # Merge visit and patient data
    if version == '2.0':
        visit_df = visit_df.rename(columns={"race": "ethnicity"})
    
    visit_pts = visit_df[[
        group_col, visit_col, admit_col, disch_col, 'los', 'los_hours',
        'admission_type', 'admission_location', 'discharge_location',
        'insurance', 'ethnicity', 'marital_status', 'hospital_expire_flag',
    ]].merge(
        pts[[group_col, 'anchor_year', 'anchor_age', 'yob', 'min_valid_year', 'dod', 'gender']],
        how='inner',
        left_on=group_col,
        right_on=group_col,
    )

    #visit_pts = visit_pts.merge(visit_df[['hadm_id', 'hospital_expire_flag']], how='left', on='hadm_id')

    # Filter for adult patients and valid years
    visit_pts['age'] = visit_pts['anchor_age']
    visit_pts = visit_pts.loc[visit_pts['age'] >= 18]
    visit_pts = visit_pts.dropna(subset=['min_valid_year'])

    return visit_pts

def partition_by_readmit(
    df: pd.DataFrame,
    gap: datetime.timedelta,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label visits based on readmission status within a specified time gap.
    
    For each visit, determines if a readmission occurred within the specified gap period.
    The gap window starts from the discharge time and considers subsequent admission times.
    
    Args:
        df: DataFrame containing visit information
        gap: Time period to consider for readmission
        group_col: Column name for patient identifier
        visit_col: Column name for visit identifier
        admit_col: Column name for admission time
        disch_col: Column name for discharge time
        
    Returns:
        Tuple containing:
        - DataFrame of visits with readmissions (case)
        - DataFrame of visits without readmissions (control)
        - DataFrame of invalid visits
    """
    case = pd.DataFrame()   # Visits with readmission within gap period
    ctrl = pd.DataFrame()   # Visits without readmission within gap period
    invalid = pd.DataFrame()    # Visits not considered in cohort
    red = pd.DataFrame()    # Readmissions from a single patient are thrown out, we're only interested in being readmitted or not in 30 days

    # Process visits grouped by patient, sorted by admission time
    grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
    
    for subject, group in tqdm(grouped):
        #print(group)
        #print('='*100)
        #print(group.shape)
        #print('='*100)
        #print(group.iloc[0])
        #print('='*100)
        #print(group.shape[0]-1)
        #print('='*100)
        if group.shape[0] <= 1:
            # Single visit - no readmission possible
            ctrl = pd.concat([ctrl, group.iloc[0]], axis=1)
            #print("To CTRL")
        else:
            for idx in range(group.shape[0]-1):
                #print(idx)
                discharge_time = group.iloc[idx][disch_col]
                #print(discharge_time)
                #print('='*100)
                # Check for readmissions within gap period
                #print(group.loc[
                #    (group[admit_col] > discharge_time) &
                #    (group[admit_col] - discharge_time <= gap)
                #])#.shape[0] >= 1
                #print('='*100)
                has_readmission = group.loc[
                    (group[admit_col] > discharge_time) &
                    (group[admit_col] - discharge_time <= gap)
                ].shape[0] >= 1
                #print(has_readmission)
                #print('='*100)
                if has_readmission and idx==0:
                    case = pd.concat([case, group.iloc[idx]], axis=1)
                    #print('**'*100)
                    #print("To CASE")
                    #print(group.iloc[idx])
                    #print('**'*100)
                elif has_readmission == False and idx==0:
                    ctrl = pd.concat([ctrl, group.iloc[idx]], axis=1)
                    #print('**'*100)
                    #print("To CTRL")
                    #print(group.iloc[idx])
                    #print('**'*100)
                else:
                    red = pd.concat([red, group.iloc[idx]], axis=1)
                    #print('**'*100)
                    #print("To REDUNDANT")
                    #print(group.iloc[idx])
                    #print('**'*100)
            
            # Last visit cannot have readmission
            #ctrl = pd.concat([ctrl, group.iloc[-1]], axis=1)

    print("[ READMISSION LABELS FINISHED ]")
    return case.T, ctrl.T, invalid.T, red.T

# def partition_by_readmit(
#     df: pd.DataFrame,
#     gap: datetime.timedelta,
#     group_col: str,
#     visit_col: str,
#     admit_col: str,
#     disch_col: str
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Label visits based on readmission status within a specified time gap.
    
#     For each visit, determines if a readmission occurred within the specified gap period.
#     The gap window starts from the discharge time and considers subsequent admission times.
    
#     Args:
#         df: DataFrame containing visit information
#         gap: Time period to consider for readmission
#         group_col: Column name for patient identifier
#         visit_col: Column name for visit identifier
#         admit_col: Column name for admission time
#         disch_col: Column name for discharge time
        
#     Returns:
#         Tuple containing:
#         - DataFrame of visits with readmissions (case)
#         - DataFrame of visits without readmissions (control)
#         - DataFrame of invalid visits
#     """
#     case = pd.DataFrame()   # Visits with readmission within gap period
#     ctrl = pd.DataFrame()   # Visits without readmission within gap period
#     invalid = pd.DataFrame()    # Visits not considered in cohort

#     # Process visits grouped by patient, sorted by admission time
#     grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
    
#     for subject, group in tqdm(grouped):
#         if group.shape[0] <= 1:
#             # Single visit - no readmission possible
#             ctrl = pd.concat([ctrl, group.iloc[0]], axis=1)
#         else:
#             for idx in range(group.shape[0]-1):
#                 visit_time = group.iloc[idx][disch_col]
                
#                 # Check for readmissions within gap period
#                 has_readmission = group.loc[
#                     (group[admit_col] > visit_time) &
#                     (group[admit_col] - visit_time <= gap)
#                 ].shape[0] >= 1
                
#                 if has_readmission:
#                     case = pd.concat([case, group.iloc[idx]], axis=1)
#                 else:
#                     ctrl = pd.concat([ctrl, group.iloc[idx]], axis=1)
            
#             # Last visit cannot have readmission
#             ctrl = pd.concat([ctrl, group.iloc[-1]], axis=1)

#     print("[ READMISSION LABELS FINISHED ]")
#     return case.T, ctrl.T, invalid.T

def get_comorbidities(diag,visits):

    diag = diag.groupby('hadm_id').count()['seq_num']

    col = [int(diag[v]) for v in visits['hadm_id'].values]

    visits['comorbs'] = col

    return visits     

def find_closest_bmi(row):
    
    patient_bmi = bmi_omr[bmi_omr['subject_id'] == row['subject_id']]
    
    if patient_bmi.empty:
        return None
    
    time_diff = abs(patient_bmi['chartdate'] - row['admittime'])
    closest_idx = time_diff.idxmin()
    
    return patient_bmi.loc[closest_idx, 'result_value']

def find_closest_bp_systolic(row):
    
    patient_bp = bp_omr[bp_omr['subject_id'] == row['subject_id']]
    
    if patient_bp.empty:
        return None
    
    time_diff = abs(patient_bp['chartdate'] - row['admittime'])
    closest_idx = time_diff.idxmin()
    
    bp = patient_bp.loc[closest_idx, 'result_value']

    systolic, diastolic = map(int, bp.split('/'))

    return systolic

def find_closest_bp_diastolic(row):
    
    patient_bp = bp_omr[bp_omr['subject_id'] == row['subject_id']]
    
    if patient_bp.empty:
        return None
    
    time_diff = abs(patient_bp['chartdate'] - row['admittime'])
    closest_idx = time_diff.idxmin()
    
    bp = patient_bp.loc[closest_idx, 'result_value']

    systolic, diastolic = map(int, bp.split('/'))

    return diastolic

def get_med_prescs(row):
    # Get all medications for a pa in a specific hospital admission
    return len(pres_df[pres_df['hadm_id'] == row['hadm_id']]) 
    
