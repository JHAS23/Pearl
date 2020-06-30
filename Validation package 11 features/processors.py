## Importing libraries


import pandas as pd
import numpy as np
import itertools
import random
import os


## Pre-processing to clean data and check data for anomalies

class Cleaning_Fetch_Data_Functions:

	## Function to replace . and , from diagnosis codes in input file
	def clean_input(self,df):
		df = df.rename(columns={'patient_id': 'patient_id', 'icd_code_type': 'type', 'icd_codes': 'codes'})
		df['codes'] = df['codes'].str.replace('.', '')
		df['codes'] = df['codes'].str.replace(',', '')
		df['codes'] = df['codes'].str.upper()
		df['type'] = df['type'].map({1:1,2:2,9:1,10:2})
		return df
		
	## Function to join ICD code wiht mapping to pull phenotypes
	def fetch_phenotype(df,phen_map):
		df= df.merge(phen_map, how = 'inner', on = ['type','codes'])
		df['feature'] = df.phenotype.str.replace('[^a-zA-Z]', '_')
		df['feature'] = df['feature'].str.lower()
		return df
	
	## Function to convert ICD9 codes to ICD10 code
	def icd_code_conversion(df,conversion_df):
		conversion_df['codes'] = conversion_df['ICD 9']
		converted_codes = pd.merge(df,conversion_df, how='left', on = 'codes')
		converted_codes= converted_codes.drop('codes',axis = 1)
		converted_codes= converted_codes.drop('type',axis = 1)
		converted_codes['type'] = 2
		converted_codes['codes'] = converted_codes['ICD 10']
		converted_codes= converted_codes.drop('ICD 10',axis = 1)
		converted_codes_final = pd.concat([df,converted_codes], ignore_index=True)
		input_df = converted_codes_final
		return input_df
	
    ## Function to create the flags fot sanity check,
    ## 1. Ony WT and HF patients cleaned and retained
	## 2. Checks to see whether there are WT patients or not
	

	def input_check(input_df):
		input_df['wt_flag'] = np.where(input_df['codes']=='E8582','1','0')
		wt_pt = input_df[input_df['wt_flag'] == '1']
		wt_pt_id = wt_pt['patient_id']
		input_df['hf_flag'] = np.where(input_df['codes']=='I504','1',np.where(input_df['codes']=='I508','1',np.where(input_df['codes']=='I5081','1',np.where(input_df['codes']=='I50','1',np.where(input_df['codes']=='I501','1',np.where(input_df['codes']=='I502','1',np.where(input_df['codes']=='I5020','1',np.where(input_df['codes']=='I5021','1',np.where(input_df['codes']=='I5022','1',np.where(input_df['codes']=='I5023','1',np.where(input_df['codes']=='I503','1',np.where(input_df['codes']=='I5030','1',np.where(input_df['codes']=='I5031','1',np.where(input_df['codes']=='I5032','1',np.where(input_df['codes']=='I5033','1',np.where(input_df['codes']=='I5040','1',np.where(input_df['codes']=='I5041','1',np.where(input_df['codes']=='I5042','1',np.where(input_df['codes']=='I5043','1',np.where(input_df['codes']=='I50810','1',np.where(input_df['codes']=='I50811','1',np.where(input_df['codes']=='I50812','1',np.where(input_df['codes']=='I50813','1',np.where(input_df['codes']=='I50814','1',np.where(input_df['codes']=='I5082','1',np.where(input_df['codes']=='I5083','1',np.where(input_df['codes']=='I5084','1',np.where(input_df['codes']=='I5089','1',np.where(input_df['codes']=='I509','1','0')))))))))))))))))))))))))))))
		input_df['any_amy'] = np.where(input_df['codes']=='E85','1',np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E851','1',np.where(input_df['codes']=='E852','1',np.where(input_df['codes']=='E853','1',np.where(input_df['codes']=='E854','1',np.where(input_df['codes']=='E858','1',np.where(input_df['codes']=='E8581','1',np.where(input_df['codes']=='E8582','1',np.where(input_df['codes']=='E8589','1',np.where(input_df['codes']=='E859','1','0')))))))))))
		hf_pt = input_df[input_df['hf_flag'] == '1']
		hf_pt = hf_pt[hf_pt['any_amy'] == '0']
		hf_pt_id = hf_pt['patient_id']
		df_cd = pd.merge(input_df,wt_pt_id, how='inner', on = 'patient_id')
		y = pd.DataFrame({'max_hf_flag' : df_cd.groupby('patient_id').max()['hf_flag']}).reset_index()
		z = pd.DataFrame({'max_hf_flag' : input_df.groupby('patient_id').max()['hf_flag']}).reset_index()
		y_0 = y[y['max_hf_flag']=='0']
		total_pat = input_df['patient_id'].nunique()
		wt_pat_count = wt_pt_id.nunique()
		hf_pat_count = hf_pt_id.nunique()
		wt_without_hf = y_0['patient_id'].nunique()
		wt_final_patient_df = y[y['max_hf_flag']!='0']
		final_patient_df = z[z['max_hf_flag']!='0']
		wt_final_pat_id = wt_final_patient_df['patient_id']
		final_pat_id= final_patient_df['patient_id']
		final_input_df = pd.merge(input_df,final_pat_id, how='inner', on = 'patient_id')
		final_input_df = final_input_df[['patient_id','codes','type','feature']]
		hf_pt_count = hf_pat_count-(wt_pat_count - wt_without_hf)
		wt_pt_count = wt_pat_count - wt_without_hf
		print("A total of " + str(total_pat) + " patients were found. Out of which we discovered " + str(wt_pat_count - wt_without_hf)  + " as Wild Type patients and a total of " + str(hf_pat_count-(wt_pat_count - wt_without_hf))  + " Heart Failure Patients.")
		print("Still processing for predictions...")
		if (wt_pat_count - wt_without_hf == 0 ):
			print("Please provide sample data for a few wild type patients as well as heart failure patients.") 
			exit(0)
		return final_input_df

potential_target_leaks = []

#Enter the name of Patient Identifier column
ptid = 'patient_id'
#Enter the name of ICD code version type column
vers_type = 'type'
#Enter the name of the ICD code column
codes = 'codes'

 
def preprocess(df,icd_map_df,ft_list_df):
   
    ## Checking for nulls
	def mis_val(df):
		#print ("Count of NA's \n",df.isnull().sum())
		nans = lambda df: df[df.isnull().any(axis=1)]
		#print ("\n Rows with NA's \r\n", nans(df))
		df_wo_na = df.dropna()
		return df_wo_na
	df_cln = mis_val(df)
	
	## Checking for ICD code and retaing only ICD9 and ICD10 codes
	def icd_fix(icd_version):
		conditions =  (icd_version.str.contains('10'))| (icd_version == '2'),(icd_version.str.contains('9'))| (icd_version == '1')
		choices = ['ICD10', 'ICD9']
		icd_fix = np.select(conditions, choices, default='NA')
		return icd_fix
    
	## Feature creation using ICD codes
	
	df_cln['type_fix'] = icd_fix(df_cln[vers_type])
	
	df_map = pd.merge(df_cln,icd_map_df,left_on=["type_fix","codes"],right_on=["type","codes"] ,how = 'left')
	df_map.to_csv('test.csv')
	df_map_cln = mis_val(df_map)
	keys = ['short_desc','major','sub_chapter']
	df_feature = df_map_cln.melt(id_vars= ptid,value_vars=keys,var_name= 'source',value_name='feature') #different syntax in lower versions
	df_feature['feature_cln'] = df_feature['feature'].str.lower().str[:100].replace(to_replace="[^A-Za-z0-9]", value="_", regex=True)
	df_feature_fitler = pd.merge(ft_list_df,df_feature,left_on = 'Feature',right_on= 'feature_cln',how= 'left')
	df_feature_cols = df_feature_fitler[[ptid,'Feature']].copy()
	df_feature_cols['presence_flag'] = '1'
	df_ft_pivot = df_feature_cols.pivot_table(index=ptid, columns='Feature', values='presence_flag', aggfunc=np.max,dropna = False,fill_value = '0')
	df_pivot_no_leaks = df_ft_pivot.drop(potential_target_leaks, axis = 1) 
	return df_pivot_no_leaks, df_ft_pivot
	
	
	## Function to mask patient ID's
def mask_patient_ids(df):
	#df['new_patient_id'] = df.reset_index().index
	#df['type'] = np.where(df['cohort_f']== 1,np.where(df['predicted_value']==1,"True Positive","False Negative"),np.where(df['predicted_value']==1,"False Positive","True Negative"))
	#patient_id_mask_mapping = df[['patient_id','new_patient_id']]
	#df['patient_id'] = df['new_patient_id']
	#df = df.drop(['new_patient_id'], axis = 1)
	return df

		
	## Function to create phenotype groups	
def process_phe_groups(df,final_input_df,final_predictions):
	df['feature'] = df.phenotype.str.replace('[^a-zA-Z]', '_')
	df['feature'] = df.feature.str.lower()
	pre = pd.merge(final_input_df,df,how = 'inner', on = 'feature')
	pre = pre[['patient_id','phenotype_group']].drop_duplicates()
	pre = pd.DataFrame(pre)
	phen_pat = pd.DataFrame({'diagnosis_combinations' : pre.groupby('patient_id').apply(lambda x: list(x.phenotype_group))}).reset_index()
	phen_pat['diagnosis_combinations'] = phen_pat['diagnosis_combinations'].astype('str').apply(lambda x:x.lower())
	phen_pat['diagnosis_combinations'] = phen_pat['diagnosis_combinations'].astype('str').str[1:-2]
	phen_pat['diagnosis_combinations'] = phen_pat['diagnosis_combinations'].astype('str').apply(lambda x:x.replace("'",""))
	phen_pat['diagnosis_combinations'] = phen_pat['diagnosis_combinations'].astype('str').apply(lambda x:x.replace(", ","|"))
	phen_pat['diagnosis_combinations'] = phen_pat.diagnosis_combinations.str.replace('[^a-zA-Z0-9|]', '_')
	phen_pat['diagnosis_combinations'] = phen_pat['diagnosis_combinations'] + '|'
	final_phenotype_comb = phen_pat.merge(final_predictions, how = 'inner', on = 'patient_id')
	final_phenotype_comb = final_phenotype_comb[['patient_id', 'diagnosis_combinations','cohort_flag']]
	final_phenotype_comb =final_phenotype_comb.drop_duplicates()
	final_phenotype_comb = final_phenotype_comb.fillna(0)
	final_phenotype_comb['cohort_type'] = np.where(final_phenotype_comb['cohort_flag']=='1','Wild Type', 'Control Group')
	final_phenotype_comb = final_phenotype_comb.drop(['cohort_flag'], axis = 1)
	return final_phenotype_comb

	
	## Function to filter conditons to create combinations
def pt_phenotype_combinations_count(input_df,string):
	input_df = input_df[input_df.cohort_type == string]
	diagnosis_list = [
	'atrial_flutter_and_or_fibrillation',
	'heart_block',
	'cardiomegaly',
	'carpal_tunnel',
	'chronic_kidney_disease',
	'joint_disorders',
	'osteoarthrosis',
	'pleurisy__pleural_effusion',
	'soft_tissue',
	'heart_failure_with_reduced_ef__systolic_or_combined_heart_failure_',
	'heart_failure_with_preserved_ef__diastolic_heart_failure_'
	]
	list_of_comb = []
	for L in range(0, 6):
		for subset in itertools.combinations(diagnosis_list, L):
			list_of_comb.append(list(subset))
	patient_count_list = []
	patient_lvl_comb = pd.DataFrame()
	for i in list_of_comb:
		text =  i
		concat_regex = ""
		concat_string = ""
		for item in text:
			concat_regex = concat_regex + "(?=.*" +  item + ")"
			concat_string = concat_string + item + "|"
		diag_comb = {}
		ct = input_df[(input_df.diagnosis_combinations.str.contains(concat_regex))].patient_id.nunique()
		pt = input_df[(input_df.diagnosis_combinations.str.contains(concat_regex))].patient_id
		pt_lvl_comb = pt.to_frame()
		pt_lvl_comb['diagnosis_combinations'] = concat_string
		diag_comb['diagnosis_combinations'] = concat_string
		diag_comb['patient_count'] = ct
		patient_count_list.append(diag_comb)
		patient_lvl_comb = patient_lvl_comb.append(pt_lvl_comb)
	patient_count_list_df= pd.DataFrame(patient_count_list)
	
	return patient_count_list_df, patient_lvl_comb
	
	
	## Function to calculate metrics on combination level (Odds ratio, counts and related metrics)
def final_processing_phe_combinations(wt_pt_ct_df,hf_pt_ct_df,desired_combinations,wt_pt_ct,hf_pt_ct):
	try:
		merged_hf_patient_count_list_df = hf_pt_ct_df.merge(desired_combinations,how = 'inner', on ='diagnosis_combinations')
		merged_hf_patient_count_list_df['count_hf_pt'] = merged_hf_patient_count_list_df['patient_count']
		merged_hf_patient_count_list_df = merged_hf_patient_count_list_df.drop('patient_count',axis=1)
		merged_patient_count_list_df = merged_hf_patient_count_list_df.merge(wt_pt_ct_df,how = 'inner', on ='diagnosis_combinations')
		merged_patient_count_list_df['count_wt_pt'] = merged_patient_count_list_df['patient_count']
		merged_patient_count_list_df = merged_patient_count_list_df.drop('patient_count',axis=1)
		merged_patient_count_list_df['TP'] =  merged_patient_count_list_df['count_wt_pt']
		merged_patient_count_list_df['FP'] =  merged_patient_count_list_df['count_hf_pt']
		merged_patient_count_list_df['TN'] =  hf_pt_ct-merged_patient_count_list_df['count_hf_pt']
		merged_patient_count_list_df['FN'] =  wt_pt_ct-merged_patient_count_list_df['count_wt_pt']
		final_merged_patient_count_list_df = merged_patient_count_list_df[['diagnosis_combinations','count_wt_pt','count_hf_pt','TP','FP','TN','FN']]
		final_merged_patient_count_list_df['Accuracy'] = (final_merged_patient_count_list_df['TP']+final_merged_patient_count_list_df['TN'])/(final_merged_patient_count_list_df['TP']+final_merged_patient_count_list_df['TN']+final_merged_patient_count_list_df['FP']+final_merged_patient_count_list_df['FN'])
		final_merged_patient_count_list_df['PPV'] = (final_merged_patient_count_list_df['TP'])/(final_merged_patient_count_list_df['TP']+final_merged_patient_count_list_df['FP'])
		final_merged_patient_count_list_df['NPV'] = (final_merged_patient_count_list_df['TN'])/(final_merged_patient_count_list_df['TN']+final_merged_patient_count_list_df['FN'])
		final_merged_patient_count_list_df['Sensitivity'] = (final_merged_patient_count_list_df['TP'])/(final_merged_patient_count_list_df['TP']+final_merged_patient_count_list_df['FN'])
		final_merged_patient_count_list_df['Specificity'] = (final_merged_patient_count_list_df['TN'])/(final_merged_patient_count_list_df['TN']+final_merged_patient_count_list_df['FP'])
		final_merged_patient_count_list_df['Odds Ratio'] = ((final_merged_patient_count_list_df['TP'])/(final_merged_patient_count_list_df['FP']))/((final_merged_patient_count_list_df['FN'])/(final_merged_patient_count_list_df['TN']))
		return final_merged_patient_count_list_df
	except:
		print("Denominator became 0")
	
		
def create_pt_lvl_comb_flags(df_wt,df_hf,desired_combinations,wt_pt,hf_pt):
	
	pat_lvl_comb = pd.DataFrame()
	pat_lvl_comb = pd.concat([df_wt,df_hf],ignore_index=True)
	pat_lvl_comb['presence_flag'] = 1
	pat_lvl_comb = pat_lvl_comb.merge(desired_combinations,how = 'inner', on ='diagnosis_combinations')
	pat_lvl_comb_pivot = pat_lvl_comb.pivot_table(index='patient_id', columns='diagnosis_combinations', values='presence_flag', aggfunc=np.max,dropna = False,fill_value = '0')
	pat_lvl_comb_pivot.reset_index(inplace = True)
	pat_lvl_comb_pivot = pd.concat([pat_lvl_comb_pivot, pd.DataFrame([[np.nan] * pat_lvl_comb_pivot.shape[1]], columns=pat_lvl_comb_pivot.columns)], ignore_index=False)
	df_wt = df_wt.merge(desired_combinations,how = 'inner', on ='diagnosis_combinations')
	df_hf = df_hf.merge(desired_combinations,how = 'inner', on ='diagnosis_combinations')
	A = df_wt.patient_id.nunique()
	B = df_hf.patient_id.nunique()
	C = hf_pt - B
	D = wt_pt - A
		
	tot_wt_pat = A
	tot_hf_pat = B
	tp = A
	fp = B
	tn = C
	fn = D
	try:
		accuracy = (tp+tn)/(tp+tn+fp+fn)
	except:
			accuracy = 0
	try:	
		precision = (tp)/(tp+fp)
	except:
		precision = 0
	try:	
		recall = (tp)/(tp+fn)
	except:
		recall = 0
	try:
		odds_ratio = ((tp)/(fp))/((fn)/(tn))
	except:
		odds_ratio = 0
	try:
		NPV = (tn)/(tn+fn)
	except:
		NPV = 0
	try:
		specificity = (tn)/(tn+fp)
	except:
		specificity = 0
	combinations_combined = {'Total WT Patients': [A], 'Total HF Patients': [B],  'True positive': [A], 'False Positive':[B], 'True Negative': [C], 'False Negative': [D], 'Accuracy':[accuracy], 'PPV':[precision],'NPV':[NPV], 'Sensitivity': [recall],'Specificity': [specificity],'Odds Ratio': [odds_ratio]}
	combinations_combined = pd.DataFrame(data=combinations_combined)
	return pat_lvl_comb_pivot,combinations_combined
	

def masking_with_mapp(df,mapping):
	#df = df.merge(mapping,how = 'inner', on ='patient_id')
	#df['patient_id'] = df['new_patient_id']
	#df = df.drop('new_patient_id', axis = 1)
	return df

def create_prediction_cohorts(df):
	input_df = df
	input_df['WT_flag'] = np.where(input_df['codes']=='E8582','1','0')
	input_df['here_flag'] = np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E852','1','0'))
	input_df['here_incl_E851_flag'] = np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E851','1',np.where(input_df['codes']=='E852','1','0')))
	wt_flag = pd.DataFrame({'max_wt_flag' : input_df.groupby('patient_id').max()['WT_flag']}).reset_index()
	here_flag = pd.DataFrame({'max_here_flag' : input_df.groupby('patient_id').max()['here_flag']}).reset_index()
	here_incl_E851_flag = pd.DataFrame({'max_here_incl_E851_flag' : input_df.groupby('patient_id').max()['here_incl_E851_flag']}).reset_index()
	input_patients=wt_flag.merge(here_flag, how = 'inner', on = 'patient_id')
	input_patients = input_patients.merge(here_incl_E851_flag, how = 'inner', on = 'patient_id')
	input_patients['hf_flag'] = np.where(input_patients['max_wt_flag']== '1','0',np.where(input_patients['max_here_flag']=='1','0',np.where(input_patients['max_here_incl_E851_flag']=='1','0','1')))
	wt_pt = input_patients[input_patients['max_wt_flag']=='1']
	here_pt = input_patients[input_patients['max_here_flag']=='1']
	here_incl_E851_pt = input_patients[input_patients['max_here_incl_E851_flag']=='1']
	hf_pt = input_patients[input_patients['hf_flag']=='1']
	total_wt_pt = wt_pt.shape[0]
	total_here_pt = here_pt.shape[0]
	total_here_incl_E851_pt = here_incl_E851_pt.shape[0]
	total_hf_pt = hf_pt.shape[0]
	random_number_list=[]
	while len(random_number_list) < total_hf_pt:
		r = random.randint(1,total_hf_pt)
		if r not in random_number_list: random_number_list.append(r)
	hf_pt['new'] = random_number_list
	hf_pt = hf_pt.sort_values('new')
	hf_pt = hf_pt.drop(columns = 'new',axis=1)
	
	if total_hf_pt<total_here_pt:
		print("Not enough Heart Failure patients for 1:1 Hereditary cohort")
		df_hered_1_1=pd.DataFrame()
	else:
		print("Creating 1:1 Hereditary cohort......")
		df_hered_1_1_p = pd.concat([here_pt,hf_pt[:total_here_pt]])
		df_hered_1_1_p = pd.DataFrame(df_hered_1_1_p['patient_id'])
		df_hered_1_1 = df_hered_1_1_p.merge(input_df, on ='patient_id', how = 'inner')
	
	if total_hf_pt<total_here_pt*9:
		print("Not enough Heart Failure patients for 1:9 Hereditary cohort")
		df_hered_1_9=pd.DataFrame()
	else:
		print("Creating 1:9 Hereditary cohort......")
		df_hered_1_9_p = pd.concat([here_pt,hf_pt[:9*total_here_pt]])
		df_hered_1_9_p = pd.DataFrame(df_hered_1_9_p['patient_id'])
		df_hered_1_9 = df_hered_1_9_p.merge(input_df, on ='patient_id', how = 'inner')
	
	if total_hf_pt<total_here_pt*19:
		print("Not enough Heart Failure patients for 1:19 Hereditary cohort")
		df_hered_1_19=pd.DataFrame()
	else:
		print("Creating 1:19 Hereditary cohort......")
		df_hered_1_19_p = pd.concat([here_pt,hf_pt[:19*total_here_pt]])
		df_hered_1_19_p = pd.DataFrame(df_hered_1_19_p['patient_id'])
		df_hered_1_19 = df_hered_1_19_p.merge(input_df, on ='patient_id', how = 'inner')
		
	if total_hf_pt<total_here_incl_E851_pt:
		print("Not enough Heart Failure patients for 1:1 Hereditary including E851 cohort")
		df_hered_incl_E851_1_1=pd.DataFrame()
	else:
		print("Creating 1:1 Hereditary including E851 cohort......")
		df_hered_incl_E851_1_1_p = pd.concat([here_incl_E851_pt,hf_pt[:total_here_incl_E851_pt]])
		df_hered_incl_E851_1_1_p = pd.DataFrame(df_hered_incl_E851_1_1_p['patient_id'])
		df_hered_incl_E851_1_1 = df_hered_incl_E851_1_1_p.merge(input_df, on ='patient_id', how = 'inner')
		
	if total_hf_pt<total_here_incl_E851_pt*9:
		print("Not enough Heart Failure patients for 1:9 Hereditary including E851 cohort")
		df_hered_incl_E851_1_9=pd.DataFrame()
	else:
		print("Creating 1:9 Hereditary including E851 cohort......")
		df_hered_incl_E851_1_9_p = pd.concat([here_incl_E851_pt,hf_pt[:9*total_here_incl_E851_pt]])
		df_hered_incl_E851_1_9_p = pd.DataFrame(df_hered_incl_E851_1_9_p['patient_id'])
		df_hered_incl_E851_1_9 = df_hered_incl_E851_1_9_p.merge(input_df, on ='patient_id', how = 'inner')
	
	if total_hf_pt<total_here_incl_E851_pt*19:
		print("Not enough Heart Failure patients for 1:1 Hereditary including E851 cohort")
		df_hered_incl_E851_1_19=pd.DataFrame()
	else:
		print("Creating 1:19 Hereditary including E851 cohort......")
		df_hered_incl_E851_1_19_p = pd.concat([here_incl_E851_pt,hf_pt[:19*total_here_incl_E851_pt]])
		df_hered_incl_E851_1_19_p = pd.DataFrame(df_hered_incl_E851_1_19_p['patient_id'])
		df_hered_incl_E851_1_19 = df_hered_incl_E851_1_19_p.merge(input_df, on ='patient_id', how = 'inner')	
		
	if total_hf_pt<total_wt_pt*19:
		print("Not enough Heart Failure patients for 1:19 Wild Type cohort")
		df_1_19=pd.DataFrame()
	else:
		print("Creating 1:19 Wild Type cohort......")
		df_1_19_p = pd.concat([wt_pt,hf_pt[:19*total_wt_pt]])
		df_1_19_p = pd.DataFrame(df_1_19_p['patient_id'])
		df_1_19 = df_1_19_p.merge(input_df, on ='patient_id', how = 'inner')
		
	if total_hf_pt<total_wt_pt*9: 
		print("Not enough Heart Failure patients for 1:9 Wild Type cohort")
		df_1_9=pd.DataFrame()
	else:
		print("Creating 1:9 Wild Type cohort......")   
		df_1_9_p = pd.concat([wt_pt,hf_pt[:9*total_wt_pt]])
		df_1_9_p = pd.DataFrame(df_1_9_p['patient_id'])
		df_1_9 = df_1_9_p.merge(input_df, on ='patient_id', how = 'inner')
	if total_hf_pt<total_wt_pt: 
		print("Not enough Heart Failure patients for 1:1 Wild Type cohort")
	else:
		print("Creating 1:1 Wild Type cohort......")  
		df_1_1_p = pd.concat([wt_pt,hf_pt[:1*total_wt_pt]])
		df_1_1_p = pd.DataFrame(df_1_1_p['patient_id'])
		df_1_1 = df_1_1_p.merge(input_df, on ='patient_id', how = 'inner')
	return df_1_1,df_1_9,df_1_19,df_hered_1_1,df_hered_1_9,df_hered_1_19,df_hered_incl_E851_1_1,df_hered_incl_E851_1_9,df_hered_incl_E851_1_19
	
	
