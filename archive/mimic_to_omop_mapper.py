import pandas as pd
import os
from datetime import datetime
from archive.omop_mapper import OMOPMapper

class MIMICToOMOPMapper(OMOPMapper):
    def __init__(self, output_dir="omop_cdm_output"):
        super().__init__(output_dir)
        self.concept_mappings = self._load_concept_mappings()
        
    def _load_concept_mappings(self):
        """Load concept mappings from CSV files in utils directory"""
        mappings = {}
        utils_dir = "utils"
        
        # Load gender mappings
        gender_map_path = os.path.join(utils_dir, "gender_mappings.csv")
        if os.path.exists(gender_map_path):
            mappings['gender'] = pd.read_csv(gender_map_path)
            
        # Load race mappings
        race_map_path = os.path.join(utils_dir, "race_mappings.csv")
        if os.path.exists(race_map_path):
            mappings['race'] = pd.read_csv(race_map_path)
            
        # Load ethnicity mappings
        ethnicity_map_path = os.path.join(utils_dir, "ethnicity_mappings.csv")
        if os.path.exists(ethnicity_map_path):
            mappings['ethnicity'] = pd.read_csv(ethnicity_map_path)
            
        return mappings
    
    def map_mimic_data(self, csv_file_path):
        """Map MIMIC-IV data to OMOP CDM format"""
        # Read the MIMIC data
        mimic_df = pd.read_csv(csv_file_path)
        
        # Map to person table
        self._map_person(mimic_df)
        
        # Map to visit_occurrence table
        self._map_visit_occurrence(mimic_df)
        
        # Map to measurement table
        self._map_measurements(mimic_df)
        
        # Map to death table
        self._map_death(mimic_df)
        
        # Save all tables
        self.save_omop_tables()
        
    def _map_person(self, mimic_df):
        """Map MIMIC data to person table"""
        person_data = []
        
        for _, row in mimic_df.iterrows():
            # Calculate birth date from anchor_year and anchor_age
            birth_year = row['anchor_year'] - row['anchor_age']
            
            person_record = {
                'person_id': row['subject_id'],
                'gender_concept_id': self._get_gender_concept_id(row['gender']),
                'year_of_birth': birth_year,
                'month_of_birth': None,  # Not available in MIMIC
                'day_of_birth': None,    # Not available in MIMIC
                'birth_datetime': f"{birth_year}-01-01",  # Approximate
                #'race_concept_id': self._get_race_concept_id(row['race']),
                'ethnicity_concept_id': self._get_ethnicity_concept_id(row['ethnicity']),
                'location_id': None,  # Not mapped
                'provider_id': None,  # Not mapped
                'care_site_id': None, # Not mapped
                'person_source_value': str(row['subject_id']),
                'gender_source_value': row['gender'],
                'gender_source_concept_id': 0,  # Not mapped
                #'race_source_value': row['race'],
                'race_source_concept_id': 0,    # Not mapped
                'ethnicity_source_value': row['ethnicity'],
                'ethnicity_source_concept_id': 0 # Not mapped
            }
            person_data.append(person_record)
            
        self.person = pd.DataFrame(person_data)
        
    def _map_visit_occurrence(self, mimic_df):
        """Map MIMIC data to visit_occurrence table"""
        visit_data = []
        
        for _, row in mimic_df.iterrows():
            visit_record = {
                'visit_occurrence_id': row['hadm_id'],
                'person_id': row['subject_id'],
                'visit_concept_id': self._get_visit_concept_id(row['admission_type']),
                'visit_start_date': row['admittime'].split()[0],
                'visit_start_datetime': row['admittime'],
                'visit_end_date': row['dischtime'].split()[0],
                'visit_end_datetime': row['dischtime'],
                'visit_type_concept_id': 44818517,  # Standard concept for 'Visit derived from EHR record'
                'provider_id': None,
                'care_site_id': None,
                'visit_source_value': str(row['hadm_id']),
                'visit_source_concept_id': 0,
                'admitted_from_concept_id': self._get_admission_location_concept_id(row['admission_location']),
                'admitted_from_source_value': row['admission_location'],
                'discharged_to_concept_id': self._get_discharge_location_concept_id(row['discharge_location']),
                'discharged_to_source_value': row['discharge_location']
            }
            visit_data.append(visit_record)
            
        self.visit_occurrence = pd.DataFrame(visit_data)
        
    def _map_measurements(self, mimic_df):
        """Map MIMIC data to measurement table"""
        measurement_data = []
        
        for _, row in mimic_df.iterrows():
            # Map BMI
            if pd.notna(row['bmi']):
                bmi_record = {
                    'measurement_id': self._generate_id(),
                    'person_id': row['subject_id'],
                    'measurement_concept_id': 3038553,  # BMI
                    'measurement_date': row['admittime'].split()[0],
                    'measurement_datetime': row['admittime'],
                    'measurement_type_concept_id': 44818701,  # From physical examination
                    'operator_concept_id': 4172703,  # Equal
                    'value_as_number': row['bmi'],
                    'value_as_concept_id': None,
                    'unit_concept_id': 9529,  # kg/m2
                    'range_low': None,
                    'range_high': None,
                    'provider_id': None,
                    'visit_occurrence_id': row['hadm_id'],
                    'visit_detail_id': None,
                    'measurement_source_value': 'BMI',
                    'measurement_source_concept_id': 0,
                    'unit_source_value': 'kg/m2',
                    'value_source_value': str(row['bmi'])
                }
                measurement_data.append(bmi_record)
            
            # Map blood pressure
            if pd.notna(row['bp_systolic']):
                bp_systolic_record = {
                    'measurement_id': self._generate_id(),
                    'person_id': row['subject_id'],
                    'measurement_concept_id': 3004249,  # Systolic blood pressure
                    'measurement_date': row['admittime'].split()[0],
                    'measurement_datetime': row['admittime'],
                    'measurement_type_concept_id': 44818701,
                    'operator_concept_id': 4172703,
                    'value_as_number': row['bp_systolic'],
                    'value_as_concept_id': None,
                    'unit_concept_id': 8876,  # mmHg
                    'range_low': None,
                    'range_high': None,
                    'provider_id': None,
                    'visit_occurrence_id': row['hadm_id'],
                    'visit_detail_id': None,
                    'measurement_source_value': 'Systolic BP',
                    'measurement_source_concept_id': 0,
                    'unit_source_value': 'mmHg',
                    'value_source_value': str(row['bp_systolic'])
                }
                measurement_data.append(bp_systolic_record)
                
            if pd.notna(row['bp_diastolic']):
                bp_diastolic_record = {
                    'measurement_id': self._generate_id(),
                    'person_id': row['subject_id'],
                    'measurement_concept_id': 3012888,  # Diastolic blood pressure
                    'measurement_date': row['admittime'].split()[0],
                    'measurement_datetime': row['admittime'],
                    'measurement_type_concept_id': 44818701,
                    'operator_concept_id': 4172703,
                    'value_as_number': row['bp_diastolic'],
                    'value_as_concept_id': None,
                    'unit_concept_id': 8876,  # mmHg
                    'range_low': None,
                    'range_high': None,
                    'provider_id': None,
                    'visit_occurrence_id': row['hadm_id'],
                    'visit_detail_id': None,
                    'measurement_source_value': 'Diastolic BP',
                    'measurement_source_concept_id': 0,
                    'unit_source_value': 'mmHg',
                    'value_source_value': str(row['bp_diastolic'])
                }
                measurement_data.append(bp_diastolic_record)
                
        self.measurement = pd.DataFrame(measurement_data)
        
    def _map_death(self, mimic_df):
        """Map MIMIC data to death table"""
        death_data = []
        
        for _, row in mimic_df.iterrows():
            if row['hospital_expire_flag'] == 1 and pd.notna(row['dod']):
                death_record = {
                    'person_id': row['subject_id'],
                    'death_date': row['dod'].split()[0],
                    'death_datetime': row['dod'],
                    'death_type_concept_id': 38003569,  # EHR record
                    'cause_concept_id': None,
                    'cause_source_value': None,
                    'cause_source_concept_id': None
                }
                death_data.append(death_record)
                
        self.death = pd.DataFrame(death_data)
        
    def _get_gender_concept_id(self, gender):
        """Map MIMIC gender to OMOP concept ID"""
        if gender == 'M':
            return 8507  # Male
        elif gender == 'F':
            return 8532  # Female
        return 0  # Unknown
        
    def _get_race_concept_id(self, race):
        """Map MIMIC race to OMOP concept ID"""
        # This would use the race mappings from utils/race_mappings.csv
        # For now, return default values
        race_map = {
            'WHITE': 8527,
            'BLACK/AFRICAN AMERICAN': 8516,
            'ASIAN': 8515,
            'HISPANIC/LATINO': 38003563
        }
        return race_map.get(race, 0)
        
    def _get_ethnicity_concept_id(self, ethnicity):
        """Map MIMIC ethnicity to OMOP concept ID"""
        # This would use the ethnicity mappings from utils/ethnicity_mappings.csv
        # For now, return default values
        ethnicity_map = {
            'HISPANIC/LATINO': 38003563,
            'NOT HISPANIC/LATINO': 38003564
        }
        return ethnicity_map.get(ethnicity, 0)
        
    def _get_visit_concept_id(self, admission_type):
        """Map MIMIC admission type to OMOP visit concept ID"""
        visit_map = {
            'EMERGENCY': 9203,  # Emergency Room Visit
            'URGENT': 9203,     # Emergency Room Visit
            'ELECTIVE': 9201,   # Inpatient Visit
            'OBSERVATION': 9202 # Outpatient Visit
        }
        return visit_map.get(admission_type, 0)
        
    def _get_admission_location_concept_id(self, location):
        """Map MIMIC admission location to OMOP concept ID"""
        location_map = {
            'EMERGENCY ROOM': 8870,
            'PHYSICIAN REFERRAL': 8870,
            'TRANSFER FROM HOSPITAL': 8863,
            'CLINIC REFERRAL': 8863
        }
        return location_map.get(location, 0)
        
    def _get_discharge_location_concept_id(self, location):
        """Map MIMIC discharge location to OMOP concept ID"""
        location_map = {
            'HOME': 8717,
            'TRANSFER TO HOSPITAL': 8717,
            'TRANSFER TO SNF': 8717,
            'DIED': 4216643
        }
        return location_map.get(location, 0) 