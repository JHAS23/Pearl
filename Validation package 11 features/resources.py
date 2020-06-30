## Importing libraries
import warnings
warnings.filterwarnings('ignore')
import sys 
import os, os.path
import pandas as pd
import numpy as np
import joblib
import re
import sklearn.metrics as metrics
import itertools



class Paths:
	## Creating paths for input files
	def __init__(self, cwd, input_file_name):
		self.input_df_path = input_file_name
		self.input_folder_path = os.path.join(cwd)
		self.icd_map_path = os.path.join(self.input_folder_path,'icd_map.csv')
		self.ft_list_path = os.path.join(self.input_folder_path,'feature_list.csv')
		self.model_file_path = os.path.join(self.input_folder_path,'rf_best_params_3_6.sav')
		self.conversion_path = os.path.join(self.input_folder_path,'icd9_to_10_conversion.csv')
		self.phenotype_mapping_path = os.path.join(self.input_folder_path, 'phenotype_mapping_processed.csv')
		self.model_file_incl_all_path = os.path.join(self.input_folder_path,'rf_best_params_3_6.sav')
		self.phenotype_groups_path = os.path.join(self.input_folder_path, 'phenotype_groups.csv')
		self.phenotype_comb_list_path = os.path.join(self.input_folder_path,'phenotype_21_combs.csv')

class Datasets:
	## Reading paths to load files onto the system
	def __init__(self, paths):
		# Datasets
		self.input_df_pre = pd.read_csv(paths.input_df_path)
		self.icd_map_df = pd.read_csv(paths.icd_map_path)
		self.ft_list_df = pd.read_csv(paths.ft_list_path)
		self.conversion_df =  pd.read_csv(paths.conversion_path)
		self.phenotype_mapping = pd.read_csv(paths.phenotype_mapping_path)
		self.model_file = joblib.load(paths.model_file_path)
		self.model_file_incl_all = joblib.load(paths.model_file_incl_all_path)
		self.phe_group = pd.read_csv(paths.phenotype_groups_path)
		self.phenotype_comb_list = pd.read_csv(paths.phenotype_comb_list_path)
	
	## Creating multiple flags at patient level for WT	
	def tag_patients(df,prediction_results,diagnosis_count,flags):
		flag_pt = pd.DataFrame()
		df['cohort_flag'] = np.where(df["codes"]== 'E8582' , '1','0')
		flag = df[df['cohort_flag'] == '1']
		flag_pt['patient_id'] = flag['patient_id']
		flag_pt = flag_pt.drop_duplicates()
		flag_pt['cohort_flag']  = '1'
		df_cd = pd.merge(flag_pt, prediction_results, how='right', on = 'patient_id')
		final = df_cd[['patient_id','cohort_flag','predicted_value','probability_of_value_0','probability_of_value_1']]
		final['cohort_f'] = np.where(final["cohort_flag"]== '1', 1,0)
		final = final.fillna(0)
		final = pd.merge(final, diagnosis_count, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E85, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E850, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E851, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E852, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E853, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E854, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E858, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E8581, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E8582, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E8589, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_E859, how='left', on = 'patient_id')
		final = pd.merge(final, flags.amy_hereditary, how='left', on = 'patient_id')
		return final

	## Creating multiple flags at patient level for here
	def tag_patients_here(df,prediction_results,diagnosis_count,flags):
		flag_pt = pd.DataFrame()
		df['cohort_flag'] = np.where(df['codes']=='E850','1',np.where(df['codes']=='E852','1','0'))
		
		flag = df[df['cohort_flag'] == '1']
		flag_pt['patient_id'] = flag['patient_id']
		flag_pt = flag_pt.drop_duplicates()
		flag_pt['cohort_flag']  = '1'
		df_cd = pd.merge(flag_pt, prediction_results, how='right', on = 'patient_id')
		final_1 = df_cd[['patient_id','cohort_flag','predicted_value','probability_of_value_0','probability_of_value_1']]
		final_1['cohort_f'] = np.where(final_1["cohort_flag"]== '1', 1,0)
		final_1 = final_1.fillna(0)
		final_1 = pd.merge(final_1, diagnosis_count, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E85, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E850, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E851, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E852, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E853, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E854, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E858, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E8581, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E8582, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E8589, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_E859, how='left', on = 'patient_id')
		final_1 = pd.merge(final_1, flags.amy_hereditary, how='left', on = 'patient_id')
		return final_1
	## Creating multiple flags at patient level for here_incl
	def tag_patients_here_incl(df,prediction_results,diagnosis_count,flags):
		flag_pt = pd.DataFrame()
		df['cohort_flag'] = np.where(df['codes']=='E850','1',np.where(df['codes']=='E851','1',np.where(df['codes']=='E852','1','0')))
		flag = df[df['cohort_flag'] == '1']
		flag_pt['patient_id'] = flag['patient_id']
		flag_pt = flag_pt.drop_duplicates()
		flag_pt['cohort_flag']  = '1'
		df_cd = pd.merge(flag_pt, prediction_results, how='right', on = 'patient_id')
		final_2 = df_cd[['patient_id','cohort_flag','predicted_value','probability_of_value_0','probability_of_value_1']]
		final_2['cohort_f'] = np.where(final_2["cohort_flag"]== '1', 1,0)
		final_2 = final_2.fillna(0)
		final_2 = pd.merge(final_2, diagnosis_count, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E85, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E850, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E851, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E852, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E853, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E854, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E858, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E8581, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E8582, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E8589, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_E859, how='left', on = 'patient_id')
		final_2 = pd.merge(final_2, flags.amy_hereditary, how='left', on = 'patient_id')
		return final_2

	
		

