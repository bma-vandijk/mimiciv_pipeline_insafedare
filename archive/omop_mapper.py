import pandas as pd
import json
import os
import datetime
import uuid
import re
from pathlib import Path

class OMOPMapper:
    """
    A class for mapping extracted medical entities and structured EHR data 
    to the OMOP Common Data Model.
    """
    
    def __init__(self, output_dir="omop_cdm_output"):
        """
        Initialize the OMOP Mapper with default tables and output directory.
        
        Args:
            output_dir (str): Directory to save OMOP CDM tables
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize empty OMOP tables
        self.person = pd.DataFrame(columns=[
            'person_id', 'gender_concept_id', 'year_of_birth', 'month_of_birth', 
            'day_of_birth', 'birth_datetime', 'race_concept_id', 'ethnicity_concept_id',
            'location_id', 'provider_id', 'care_site_id', 'person_source_value',
            'gender_source_value', 'gender_source_concept_id', 'race_source_value',
            'race_source_concept_id', 'ethnicity_source_value', 'ethnicity_source_concept_id'
        ])
        
        self.condition_occurrence = pd.DataFrame(columns=[
            'condition_occurrence_id', 'person_id', 'condition_concept_id', 
            'condition_start_date', 'condition_start_datetime', 'condition_end_date',
            'condition_end_datetime', 'condition_type_concept_id', 'condition_status_concept_id',
            'stop_reason', 'provider_id', 'visit_occurrence_id', 'visit_detail_id',
            'condition_source_value', 'condition_source_concept_id', 'condition_status_source_value'
        ])
        
        self.measurement = pd.DataFrame(columns=[
            'measurement_id', 'person_id', 'measurement_concept_id', 'measurement_date',
            'measurement_datetime', 'measurement_time', 'measurement_type_concept_id',
            'operator_concept_id', 'value_as_number', 'value_as_concept_id',
            'unit_concept_id', 'range_low', 'range_high', 'provider_id',
            'visit_occurrence_id', 'visit_detail_id', 'measurement_source_value',
            'measurement_source_concept_id', 'unit_source_value', 'value_source_value'
        ])
        
        self.drug_exposure = pd.DataFrame(columns=[
            'drug_exposure_id', 'person_id', 'drug_concept_id', 'drug_exposure_start_date',
            'drug_exposure_start_datetime', 'drug_exposure_end_date', 'drug_exposure_end_datetime',
            'verbatim_end_date', 'drug_type_concept_id', 'stop_reason', 'refills',
            'quantity', 'days_supply', 'sig', 'route_concept_id', 'lot_number',
            'provider_id', 'visit_occurrence_id', 'visit_detail_id', 'drug_source_value',
            'drug_source_concept_id', 'route_source_value', 'dose_unit_source_value'
        ])
        
        self.observation = pd.DataFrame(columns=[
            'observation_id', 'person_id', 'observation_concept_id', 'observation_date',
            'observation_datetime', 'observation_type_concept_id', 'value_as_number',
            'value_as_string', 'value_as_concept_id', 'qualifier_concept_id',
            'unit_concept_id', 'provider_id', 'visit_occurrence_id', 'visit_detail_id',
            'observation_source_value', 'observation_source_concept_id', 'unit_source_value',
            'qualifier_source_value'
        ])
        
        self.visit_occurrence = pd.DataFrame(columns=[
            'visit_occurrence_id', 'person_id', 'visit_concept_id', 'visit_start_date',
            'visit_start_datetime', 'visit_end_date', 'visit_end_datetime', 'visit_type_concept_id',
            'provider_id', 'care_site_id', 'visit_source_value', 'visit_source_concept_id',
            'admitted_from_concept_id', 'admitted_from_source_value', 'discharge_to_concept_id',
            'discharge_to_source_value', 'preceding_visit_occurrence_id'
        ])
        
        # Add NOTE_NLP table to store NLP output from clinical notes
        self.note_nlp = pd.DataFrame(columns=[
            'note_nlp_id', 'note_id', 'section_concept_id', 'snippet', 
            'offset', 'lexical_variant', 'note_nlp_concept_id', 'note_nlp_source_concept_id',
            'nlp_system', 'nlp_date', 'nlp_datetime', 'term_exists', 
            'term_temporal', 'term_modifiers', 'note_nlp_source_value'
        ])
    
    def _generate_id(self):
        """Generate a unique ID for OMOP records"""
        return abs(hash(str(uuid.uuid4())))
    
    def save_omop_tables(self):
        """
        Save all OMOP tables to CSV files in the output directory.
        
        Returns:
            dict: Dictionary with paths to saved CSV files
        """
        tables = {
            'person': self.person,
            'condition_occurrence': self.condition_occurrence,
            'measurement': self.measurement,
            'drug_exposure': self.drug_exposure,
            'observation': self.observation,
            'visit_occurrence': self.visit_occurrence,
            'note_nlp': self.note_nlp
        }
        
        file_paths = {}
        
        for table_name, df in tables.items():
            if not df.empty:
                file_path = os.path.join(self.output_dir, f"{table_name}.csv")
                df.to_csv(file_path, index=False)
                file_paths[table_name] = file_path
                print(f"Saved {table_name} with {len(df)} records to {file_path}")
        
        return file_paths