## Importing subset files for inputs and processing

import resources as res
import processors as pr
import calculations as calc
import code_reduction as red
import os, os.path, shutil
import pandas as pd


# Loading input patient level file for prediction
#input_dir = os.path.join(os.getcwd().strip(),"Inputs")
input_dir = os.path.join(os.getcwd().strip())
parent_dir = input_dir[0:-6]
#print("Please make sure to copy over input csv file to the location " + input_dir)
#print("Make sure that your input file follows the same format as sample_input.csv")
input_name = "cleaned_input.csv"
input_file_name = os.path.join(input_dir,input_name.strip())
demographics_file_name = input("Input Demographics File Name (example: sample_input.csv): ")
demographics_file_path = os.path.join(input_dir,"user_input_files")
demographics_file_path = os.path.join(demographics_file_path,demographics_file_name.strip())
demographics_file = pd.read_csv(demographics_file_path)
prob_adjst = "prob_adjustment.csv"
prob_adjst_path = os.path.join(input_dir,prob_adjst.strip())
prob_adjst_file = pd.read_csv(prob_adjst_path)

## Create input path using file name and current working directory
input_paths = res.Paths(os.getcwd().strip(),input_file_name.strip())

## Calling input files using path created in previous step
input_files = res.Datasets(input_paths)


## Cleaning the input file
print("Cleaning input file...")
input_df_pre = pr.Cleaning_Fetch_Data_Functions()
input_df_pre = input_df_pre.clean_input(input_files.input_df_pre)
input_df_pre = pr.Cleaning_Fetch_Data_Functions.fetch_phenotype(input_df_pre,input_files.phenotype_mapping)
input_df = pr.Cleaning_Fetch_Data_Functions.icd_code_conversion(input_df_pre,input_files.conversion_df)
input_df = input_df.drop_duplicates()

## Count of Diagnosis code at Patient level
diagnosis_count = res.pd.DataFrame({'diag_count' : input_df.groupby('patient_id').count().codes}).reset_index()

## Creating flags for amyloidosis codes at patient level
flags = calc.Flags(input_df)

## Checking the sanity of the input data wrt HF and WT patients
final_input_df = pr.Cleaning_Fetch_Data_Functions.input_check(input_df)

#final_input_df = final_input_df.merge(input_files.icd_map_df['codes'], on = 'codes', how = 'inner')

## Create Prediction Cohorts
df_1_1,df_1_9,df_1_19,df_hered_1_1,df_hered_1_9,df_hered_1_19,df_hered_incl_E851_1_1,df_hered_incl_E851_1_9,df_hered_incl_E851_1_19 = pr.create_prediction_cohorts(final_input_df)


## Converting the datatype of ICD codes and type to string
dtype1 = dict(codes = str, type = str)


# Determine which models to run based on cohorts made from input data (if empty don't run)
run_wt_1_1 = not df_1_1.empty
run_wt_1_9 = not df_1_9.empty
run_wt_1_19 = not df_1_19.empty
run_hered_1_1 = not df_hered_1_1.empty
run_hered_incl_E851_1_1 = not df_hered_incl_E851_1_1.empty
run_hered_1_9 = not df_hered_1_9.empty
run_hered_incl_E851_1_9 = not df_hered_incl_E851_1_9.empty
run_hered_1_19 = not df_hered_1_19.empty
run_hered_incl_E851_1_19 = not df_hered_incl_E851_1_19.empty



# Preprocess
final_df_excluding_1_cm, final_df_including_all_features = pr.preprocess(final_input_df.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_wt_1_1:
	final_df_excluding_1_cm_df_1_1, final_df_including_all_features_df_1_1 = pr.preprocess(df_1_1.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_wt_1_9:
	final_df_excluding_1_cm_df_1_9, final_df_including_all_features_df_1_9 = pr.preprocess(df_1_9.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_wt_1_19:
	final_df_excluding_1_cm_df_1_19, final_df_including_all_features_df_1_19 = pr.preprocess(df_1_19.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_1_1:
	final_df_excluding_1_cm_df_hered_1_1, final_df_including_all_features_df_hered_1_1 = pr.preprocess(df_hered_1_1.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_1_9:
	final_df_excluding_1_cm_df_hered_1_9, final_df_including_all_features_df_hered_1_9 = pr.preprocess(df_hered_1_9.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_1_19:
	final_df_excluding_1_cm_df_hered_1_19, final_df_including_all_features_df_hered_1_19 = pr.preprocess(df_hered_1_19.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_incl_E851_1_1:
	final_df_excluding_1_cm_df_hered_incl_E851_1_1, final_df_including_all_features_df_hered_incl_E851_1_1 = pr.preprocess(df_hered_incl_E851_1_1.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_incl_E851_1_9:
	final_df_excluding_1_cm_df_hered_incl_E851_1_9, final_df_including_all_features_df_hered_incl_E851_1_9 = pr.preprocess(df_hered_incl_E851_1_9.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)
if run_hered_incl_E851_1_19:
	final_df_excluding_1_cm_df_hered_incl_E851_1_19, final_df_including_all_features_df_hered_incl_E851_1_19 = pr.preprocess(df_hered_incl_E851_1_19.astype(dtype1), input_files.icd_map_df,input_files.ft_list_df)

	
## Prediction for both the scenarios- 
	#1)Excluding Cardiomyopathy in diseases classified elsewhere
	#2)Including all the features
print("Creating predictions with cleaned data...")



##wt df_1_1
if run_wt_1_1:
	prediction_results_excluding_1_cm_df_1_1 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_1_1)
	prediction_results_including_all_features_df_1_1 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_1_1)
	final_dataset_excluding_1_cm_df_1_1 = res.Datasets.tag_patients(df_1_1,prediction_results_excluding_1_cm_df_1_1,diagnosis_count,flags)
	final_dataset_including_all_features_df_1_1 = res.Datasets.tag_patients(df_1_1,prediction_results_including_all_features_df_1_1,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_1_1 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_1_1,final_dataset_excluding_1_cm_df_1_1['cohort_f'])
	model_metrics_including_all_features_df_1_1 = calc.custom_model_metrics(final_dataset_including_all_features_df_1_1,final_dataset_including_all_features_df_1_1['cohort_f'])

##wt df_1_9
if run_wt_1_9:
	prediction_results_excluding_1_cm_df_1_9 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_1_9)
	prediction_results_including_all_features_df_1_9 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_1_9)
	final_dataset_excluding_1_cm_df_1_9 = res.Datasets.tag_patients(df_1_9,prediction_results_excluding_1_cm_df_1_9,diagnosis_count,flags)
	final_dataset_including_all_features_df_1_9 = res.Datasets.tag_patients(df_1_9,prediction_results_including_all_features_df_1_9,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_1_9 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_1_9,final_dataset_excluding_1_cm_df_1_9['cohort_f'])
	model_metrics_including_all_features_df_1_9 = calc.custom_model_metrics(final_dataset_including_all_features_df_1_9,final_dataset_including_all_features_df_1_9['cohort_f'])

##wt df_1_19

if run_wt_1_19:
	prediction_results_excluding_1_cm_df_1_19 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_1_19)
	prediction_results_including_all_features_df_1_19 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_1_19)
	final_dataset_excluding_1_cm_df_1_19 = res.Datasets.tag_patients(df_1_19,prediction_results_excluding_1_cm_df_1_19,diagnosis_count,flags)
	
	final_dataset_including_all_features_df_1_19 = res.Datasets.tag_patients(df_1_19,prediction_results_including_all_features_df_1_19,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_1_19 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_1_19,final_dataset_excluding_1_cm_df_1_19['cohort_f'])
	model_metrics_including_all_features_df_1_19 = calc.custom_model_metrics(final_dataset_including_all_features_df_1_19,final_dataset_including_all_features_df_1_19['cohort_f'])

#hereditary df_1_1
if run_hered_1_1:
	prediction_results_excluding_1_cm_df_hered_1_1 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_1_1)
	prediction_results_including_all_features_df_hered_1_1 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_1_1)
	final_dataset_excluding_1_cm_df_hered_1_1 = res.Datasets.tag_patients_here(df_hered_1_1,prediction_results_excluding_1_cm_df_hered_1_1,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_1_1 = res.Datasets.tag_patients_here(df_hered_1_1,prediction_results_including_all_features_df_hered_1_1,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_1_1 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_1_1,final_dataset_excluding_1_cm_df_hered_1_1['cohort_f'])
	model_metrics_including_all_features_df_hered_1_1 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_1_1,final_dataset_including_all_features_df_hered_1_1['cohort_f'])

#hereditary df_1_9
if run_hered_1_9:
	prediction_results_excluding_1_cm_df_hered_1_9 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_1_9)
	prediction_results_including_all_features_df_hered_1_9 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_1_9)
	final_dataset_excluding_1_cm_df_hered_1_9 = res.Datasets.tag_patients_here(df_hered_1_9,prediction_results_excluding_1_cm_df_hered_1_9,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_1_9 = res.Datasets.tag_patients_here(df_hered_1_9,prediction_results_including_all_features_df_hered_1_9,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_1_9 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_1_9,final_dataset_excluding_1_cm_df_hered_1_9['cohort_f'])
	model_metrics_including_all_features_df_hered_1_9 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_1_9,final_dataset_including_all_features_df_hered_1_9['cohort_f'])

#hereditary df_1_19
if run_hered_1_19:
	prediction_results_excluding_1_cm_df_hered_1_19 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_1_19)
	prediction_results_including_all_features_df_hered_1_19 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_1_19)
	final_dataset_excluding_1_cm_df_hered_1_19 = res.Datasets.tag_patients_here(df_hered_1_19,prediction_results_excluding_1_cm_df_hered_1_19,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_1_19 = res.Datasets.tag_patients_here(df_hered_1_19,prediction_results_including_all_features_df_hered_1_19,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_1_19 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_1_19,final_dataset_excluding_1_cm_df_hered_1_19['cohort_f'])
	model_metrics_including_all_features_df_hered_1_19 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_1_19,final_dataset_including_all_features_df_hered_1_19['cohort_f'])

	
#hereditary including E851 df_1_1
if run_hered_incl_E851_1_1:
	prediction_results_excluding_1_cm_df_hered_incl_E851_1_1 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_incl_E851_1_1)
	prediction_results_including_all_features_df_hered_incl_E851_1_1 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_incl_E851_1_1)
	final_dataset_excluding_1_cm_df_hered_incl_E851_1_1 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_1,prediction_results_excluding_1_cm_df_hered_incl_E851_1_1,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_incl_E851_1_1 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_1,prediction_results_including_all_features_df_hered_incl_E851_1_1,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_incl_E851_1_1 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_incl_E851_1_1,final_dataset_excluding_1_cm_df_hered_incl_E851_1_1['cohort_f'])
	model_metrics_including_all_features_df_hered_incl_E851_1_1 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_incl_E851_1_1,final_dataset_including_all_features_df_hered_incl_E851_1_1['cohort_f'])

#hereditary including E851 df_1_9
if run_hered_incl_E851_1_9:
	prediction_results_excluding_1_cm_df_hered_incl_E851_1_9 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_incl_E851_1_9)
	prediction_results_including_all_features_df_hered_incl_E851_1_9 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_incl_E851_1_9)
	final_dataset_excluding_1_cm_df_hered_incl_E851_1_9 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_9,prediction_results_excluding_1_cm_df_hered_incl_E851_1_9,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_incl_E851_1_9 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_9,prediction_results_including_all_features_df_hered_incl_E851_1_9,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_incl_E851_1_9 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_incl_E851_1_9,final_dataset_excluding_1_cm_df_hered_incl_E851_1_9['cohort_f'])
	model_metrics_including_all_features_df_hered_incl_E851_1_9 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_incl_E851_1_9,final_dataset_including_all_features_df_hered_incl_E851_1_9['cohort_f'])

#hereditary including E851 df_1_19
if run_hered_incl_E851_1_19:
	prediction_results_excluding_1_cm_df_hered_incl_E851_1_19 = calc.predictions(input_files.model_file,final_df_excluding_1_cm_df_hered_incl_E851_1_19)
	prediction_results_including_all_features_df_hered_incl_E851_1_19 = calc.predictions(input_files.model_file_incl_all,final_df_including_all_features_df_hered_incl_E851_1_19)
	final_dataset_excluding_1_cm_df_hered_incl_E851_1_19 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_19,prediction_results_excluding_1_cm_df_hered_incl_E851_1_19,diagnosis_count,flags)
	final_dataset_including_all_features_df_hered_incl_E851_1_19 = res.Datasets.tag_patients_here_incl(df_hered_incl_E851_1_19,prediction_results_including_all_features_df_hered_incl_E851_1_19,diagnosis_count,flags)
	model_metrics_excluding_1_cm_df_hered_incl_E851_1_19 = calc.custom_model_metrics(final_dataset_excluding_1_cm_df_hered_incl_E851_1_19,final_dataset_excluding_1_cm_df_hered_incl_E851_1_19['cohort_f'])
	model_metrics_including_all_features_df_hered_incl_E851_1_19 = calc.custom_model_metrics(final_dataset_including_all_features_df_hered_incl_E851_1_19,final_dataset_including_all_features_df_hered_incl_E851_1_19['cohort_f'])
	

## Counting the number of WT and HF patients
if run_wt_1_1:
	wt_pt_ct = res.pd.to_numeric(final_dataset_excluding_1_cm_df_1_1.cohort_flag).sum()
	hf_pt_ct = final_dataset_excluding_1_cm_df_1_1.patient_id.nunique() - wt_pt_ct
else:
	# if df_1_1 is empty then there are 0 wt or hf patients
	wt_pt_ct = 0 
	hf_pt_ct = 0

## Grouping the Phenotypes and calculating metrics for phenotype combination
if run_wt_1_1:
	phe_comb_input = pr.process_phe_groups(input_files.phe_group,df_1_1,final_dataset_excluding_1_cm_df_1_1)
	wt_pt_phe_combination_counts, wt_pat_lvl_comb = pr.pt_phenotype_combinations_count(phe_comb_input,'Wild Type')
	hf_pt_phe_combination_counts, hf_pat_lvl_comb = pr.pt_phenotype_combinations_count(phe_comb_input,'Control Group')
	phenotype_combinations_comparison = pr.final_processing_phe_combinations(wt_pt_phe_combination_counts,hf_pt_phe_combination_counts,input_files.phenotype_comb_list,wt_pt_ct,hf_pt_ct)


##Patient ID Masking
prediction_results_excluding_1_cm = calc.predictions(input_files.model_file,final_df_excluding_1_cm) 
if run_wt_1_1:
	final_dataset_excluding_1_cm = res.Datasets.tag_patients(df_1_1,prediction_results_excluding_1_cm,diagnosis_count,flags)
	mask_mapping = pr.mask_patient_ids(final_dataset_excluding_1_cm)
	final_prediction_output_excluding_1_cm_df_1_1 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_1,mask_mapping)
	final_prediction_output_including_all_features_df_1_1 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_1,mask_mapping)

if run_wt_1_9:
	final_prediction_output_excluding_1_cm_df_1_9 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_9,mask_mapping)
	final_prediction_output_including_all_features_df_1_9 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_9,mask_mapping)

if run_wt_1_19:
	final_prediction_output_excluding_1_cm_df_1_19 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_19,mask_mapping)
	final_prediction_output_including_all_features_df_1_19 = pr.masking_with_mapp(final_dataset_excluding_1_cm_df_1_19,mask_mapping)

if run_hered_1_1:
	final_hered_prediction_output_excluding_1_cm_df_1_1 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_1,mask_mapping)
	final_hered_prediction_output_including_all_features_df_1_1 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_1,mask_mapping)
	

if run_hered_1_9:
	final_hered_prediction_output_excluding_1_cm_df_1_9 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_9,mask_mapping)
	final_hered_prediction_output_including_all_features_df_1_9 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_9,mask_mapping)
		

if run_hered_1_19:
	final_hered_prediction_output_excluding_1_cm_df_1_19 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_19,mask_mapping)
	final_hered_prediction_output_including_all_features_df_1_19 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_1_19,mask_mapping)
		
	
if run_hered_incl_E851_1_1:
	final_hered_incl_prediction_output_excluding_1_cm_df_1_1 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_1,mask_mapping)
	final_hered_incl_prediction_output_including_all_features_df_1_1 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_1,mask_mapping)
	

if run_hered_incl_E851_1_9:
	final_hered_incl_prediction_output_excluding_1_cm_df_1_9 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_9,mask_mapping)
	final_hered_incl_prediction_output_including_all_features_df_1_9 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_9,mask_mapping)
		

if run_hered_incl_E851_1_19:
	final_hered_incl_prediction_output_excluding_1_cm_df_1_19 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_19,mask_mapping)
	final_hered_incl_prediction_output_including_all_features_df_1_19 = pr.masking_with_mapp(final_dataset_including_all_features_df_hered_incl_E851_1_19,mask_mapping)
		

## Patient Level Phenotype Combinations Flag
#pat_lvl_comb_pivot,combinations_combined = pr.create_pt_lvl_comb_flags(wt_pat_lvl_comb,hf_pat_lvl_comb,input_files.phenotype_comb_list,wt_pt_ct,hf_pt_ct)
#final_pat_lvl_comb_pivot = pr.masking_with_mapp(pat_lvl_comb_pivot,mask_mapping)




## Exporting the Outputs for both the scenarios-
	#1)Excluding Cardiomyopathy in diseases classified elsewhere
	#2)Including all the features
output_dir = os.path.join(parent_dir.strip(),'Outputs')

def create_output_folder(path):
	# Will delete folder if folder already exists
	try:
		if os.path.exists(path):
			shutil.rmtree(path)
		os.mkdir(path)
	except OSError:
		print("Creation of the directory {} failed".format(path))

if run_wt_1_1:	
	final_prediction_output_including_all_features_df_1_1 = final_prediction_output_including_all_features_df_1_1.merge(demographics_file, on = 'patient_id', how = 'left')
	final_prediction_output_including_all_features_df_1_1 = final_prediction_output_including_all_features_df_1_1.merge(prob_adjst_file, on = ['age','sex'], how = 'left')
	final_prediction_output_including_all_features_df_1_1['adjusted_probability'] = final_prediction_output_including_all_features_df_1_1['multiplier']*final_prediction_output_including_all_features_df_1_1['probability_of_value_1']
	final_prediction_output_including_all_features_df_1_1['suspicion_index'] = final_prediction_output_including_all_features_df_1_1['probability_of_value_1']/final_prediction_output_including_all_features_df_1_1['probability_of_value_0']
	final_prediction_output_including_all_features_df_1_1 = final_prediction_output_including_all_features_df_1_1.drop(['age_sex_wt_count','wt_posterior_age_sex','hf_posterior_age_sex','age_sex_hf_count','multiplier'],axis =1)

if run_wt_1_9:
	final_prediction_output_including_all_features_df_1_9 = final_prediction_output_including_all_features_df_1_9.merge(demographics_file, on = 'patient_id', how = 'left')
	final_prediction_output_including_all_features_df_1_9 = final_prediction_output_including_all_features_df_1_9.merge(prob_adjst_file, on = ['age','sex'], how = 'left')
	final_prediction_output_including_all_features_df_1_9['adjusted_probability'] = final_prediction_output_including_all_features_df_1_9['multiplier']*final_prediction_output_including_all_features_df_1_9['probability_of_value_1']
	final_prediction_output_including_all_features_df_1_9['suspicion_index'] = final_prediction_output_including_all_features_df_1_9['probability_of_value_1']/final_prediction_output_including_all_features_df_1_9['probability_of_value_0']
	final_prediction_output_including_all_features_df_1_9 = final_prediction_output_including_all_features_df_1_9.drop(['age_sex_wt_count','wt_posterior_age_sex','hf_posterior_age_sex','age_sex_hf_count','multiplier'],axis =1)

if run_wt_1_19:
	final_prediction_output_including_all_features_df_1_19 = final_prediction_output_including_all_features_df_1_19.merge(demographics_file, on = 'patient_id', how = 'left')
	final_prediction_output_including_all_features_df_1_19 = final_prediction_output_including_all_features_df_1_19.merge(prob_adjst_file, on = ['age','sex'], how = 'left')
	final_prediction_output_including_all_features_df_1_19['adjusted_probability'] = final_prediction_output_including_all_features_df_1_19['multiplier']*final_prediction_output_including_all_features_df_1_19['probability_of_value_1']
	final_prediction_output_including_all_features_df_1_19['suspicion_index'] = final_prediction_output_including_all_features_df_1_19['probability_of_value_1']/final_prediction_output_including_all_features_df_1_19['probability_of_value_0']
	final_prediction_output_including_all_features_df_1_19 = final_prediction_output_including_all_features_df_1_19.drop(['age_sex_wt_count','wt_posterior_age_sex','hf_posterior_age_sex','age_sex_hf_count','multiplier'],axis =1)

if run_wt_1_1:
	print("Exporting output files for WT 1:1...")
	wt_1_1_path = os.path.join(output_dir,"WT_1_1")
	create_output_folder(wt_1_1_path)
	# change directory
	os.chdir(wt_1_1_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_1.output_matrix_2).to_csv('model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_1.output_matrix).to_csv('model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_1.output_matrix_2).to_csv('model_metrics_including_all_features.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_1.output_matrix).to_csv('model_metrics_including_all_features.csv',mode='a', header=False, index = False)
	#res.pd.DataFrame(final_prediction_output_excluding_1_cm_df_1_1).to_csv('final_prediction_output_excluding_1_cm.csv')
	res.pd.DataFrame(final_prediction_output_including_all_features_df_1_1).to_csv('final_prediction_output_including_all_features.csv')
	#res.pd.DataFrame(phenotype_combinations_comparison).to_csv('phenotype_combinations_comparison.csv')
	#res.pd.DataFrame(final_pat_lvl_comb_pivot).to_csv('patient_level_phenotype_combinations.csv',header = True, index  = False)
	#res.pd.DataFrame(combinations_combined).to_csv('patient_level_phenotype_combinations.csv',mode='a',header = True, index  = False)

if run_wt_1_9:
	print("Exporting output files for WT 1:9...")
	wt_1_9_path = os.path.join(output_dir,"WT_1_9")
	create_output_folder(wt_1_9_path)
	# change directory	
	os.chdir(wt_1_9_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_9.output_matrix_2).to_csv('model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_9.output_matrix).to_csv('model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_9.output_matrix_2).to_csv('model_metrics_including_all_features.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_9.output_matrix).to_csv('model_metrics_including_all_features.csv',mode='a', header=False, index = False)
	#res.pd.DataFrame(final_prediction_output_excluding_1_cm_df_1_9).to_csv('final_prediction_output_excluding_1_cm.csv')
	res.pd.DataFrame(final_prediction_output_including_all_features_df_1_9).to_csv('final_prediction_output_including_all_features.csv')

if run_wt_1_19:
	print("Exporting output files for WT 1:19...")
	wt_1_19_path = os.path.join(output_dir,"WT_1_19")
	create_output_folder(wt_1_19_path)
	# change directory
	os.chdir(wt_1_19_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_19.output_matrix_2).to_csv('model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_1_19.output_matrix).to_csv('model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_19.output_matrix_2).to_csv('model_metrics_including_all_features.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_1_19.output_matrix).to_csv('model_metrics_including_all_features.csv',mode='a', header=False, index = False)
	#res.pd.DataFrame(final_prediction_output_excluding_1_cm_df_1_19).to_csv('final_prediction_output_excluding_1_cm.csv')
	res.pd.DataFrame(final_prediction_output_including_all_features_df_1_19).to_csv('final_prediction_output_including_all_features.csv')

if run_hered_1_1:
	
	print("Exporting output files for Herediatry Patients 1:1...")
	hereditary_1_1_path = res.os.path.join(output_dir,"Hereditary_1_1")
	#print(hereditary_1_1_path)
	create_output_folder(hereditary_1_1_path)
	# change directory
	os.chdir(hereditary_1_1_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_1.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_1.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_1.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_1.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(final_hered_prediction_output_excluding_1_cm_df_1_1).to_csv('final_hered_prediction_output_including_all_features.csv')
	
if run_hered_1_9:
	print("Exporting output files for Herediatry Patients 1:9...")
	hereditary_1_9_path = res.os.path.join(output_dir,"Hereditary_1_9")
	create_output_folder(hereditary_1_9_path)
	# change directory
	os.chdir(hereditary_1_9_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_9.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_9.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_9.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_9.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(final_hered_prediction_output_excluding_1_cm_df_1_9).to_csv('final_hered_prediction_output_including_all_features.csv')
	
if run_hered_1_19:
	print("Exporting output files for Herediatry Patients 1:19...")
	hereditary_1_19_path = res.os.path.join(output_dir,"Hereditary_1_19")
	create_output_folder(hereditary_1_19_path)
	# change directory
	os.chdir(hereditary_1_19_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_19.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_1_19.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_19.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_1_19.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)	
	res.pd.DataFrame(final_hered_prediction_output_excluding_1_cm_df_1_19).to_csv('final_hered_prediction_output_including_all_features.csv')
	
if run_hered_incl_E851_1_1:
	print("Exporting output files for Herediatry Including E851 Patients 1:1...")
	hereditary_incl_E851_1_1_path = res.os.path.join(output_dir,"Hereditary_Including_E851_1_1")
	create_output_folder(hereditary_incl_E851_1_1_path)
	# change directory
	os.chdir(hereditary_incl_E851_1_1_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_1.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_1.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_1.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_1.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(final_hered_incl_prediction_output_including_all_features_df_1_1).to_csv('final_hered_incl_prediction_output_including_all_features.csv')
	
if run_hered_incl_E851_1_9:
	print("Exporting output files for Herediatry Including E851 Patients 1:9...")
	hereditary_incl_E851_1_9_path = res.os.path.join(output_dir,"Hereditary_Including_E851_1_9")
	create_output_folder(hereditary_incl_E851_1_9_path)
	# change directory
	os.chdir(hereditary_incl_E851_1_9_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_9.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_9.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_9.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_9.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(final_hered_incl_prediction_output_including_all_features_df_1_9).to_csv('final_hered_incl_prediction_output_including_all_features.csv')
	
if run_hered_incl_E851_1_19:
	print("Exporting output files for Herediatry Including E851 Patients 1:19...")
	hereditary_incl_E851_1_19_path = res.os.path.join(output_dir,"Hereditary_Including_E851_1_19")
	create_output_folder(hereditary_incl_E851_1_19_path)
	# change directory
	os.chdir(hereditary_incl_E851_1_19_path)

	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_19.output_matrix_2).to_csv('hereditary_model_metrics_excluding_1_cm.csv',header = False, index  = False)
	#res.pd.DataFrame(model_metrics_excluding_1_cm_df_hered_incl_E851_1_19.output_matrix).to_csv('hereditary_model_metrics_excluding_1_cm.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_19.output_matrix_2).to_csv('hereditary_model_metrics_including_all_feature.csv',header = False, index  = False)
	res.pd.DataFrame(model_metrics_including_all_features_df_hered_incl_E851_1_19.output_matrix).to_csv('hereditary_model_metrics_including_all_feature.csv',mode='a', header=False, index = False)
	res.pd.DataFrame(final_hered_incl_prediction_output_including_all_features_df_1_19).to_csv('final_hered_incl_prediction_output_including_all_features.csv')	

print("Process finished! Check output folder for results here: " + output_dir)
