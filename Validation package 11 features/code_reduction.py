import pandas as pd
import os
import numpy as np
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_dir = os.path.join(os.getcwd().strip(),"Inputs")
#print("Make sure that your input file is present in the same folder from where the python is launched")
input_name = input("Input File Name (example: sample_input.csv): ")
input_path = os.path.join(input_dir.strip(), input_name)
input_file = pd.read_csv(input_path)

print("Cleaning input file...")
input_file = input_file.drop_duplicates()
input_file['codes'] = input_file['codes'].str.replace('.', '')
input_file['codes'] = input_file['codes'].str.replace(',', '')
input_file['codes'] = input_file['codes'].str.upper()
conversion_file_path = os.path.join(input_dir,'icd9_to_10_conversion.csv')
conversion_file = pd.read_csv(conversion_file_path)
conversion_file['codes'] = conversion_file['ICD 9']
input_file_9 = input_file[input_file['type']==1]
input_file_10 = input_file[input_file['type']==2]
input_file_10 = input_file_10[['patient_id','codes']]
converted_codes = pd.merge(input_file_9,conversion_file, how='inner', on = 'codes')
converted_codes['codes'] = converted_codes['ICD 10']
converted_codes = converted_codes[['patient_id','codes']]
converted_df = pd.concat([converted_codes,input_file_10])

input_df = converted_df
input_df['WT_flag'] = np.where(input_df['codes']=='E8582','1','0')
input_df['hf_flag'] = np.where(input_df['codes']=='I504','1',np.where(input_df['codes']=='I508','1',np.where(input_df['codes']=='I5081','1',np.where(input_df['codes']=='I50','1',np.where(input_df['codes']=='I501','1',np.where(input_df['codes']=='I502','1',np.where(input_df['codes']=='I5020','1',np.where(input_df['codes']=='I5021','1',np.where(input_df['codes']=='I5022','1',np.where(input_df['codes']=='I5023','1',np.where(input_df['codes']=='I503','1',np.where(input_df['codes']=='I5030','1',np.where(input_df['codes']=='I5031','1',np.where(input_df['codes']=='I5032','1',np.where(input_df['codes']=='I5033','1',np.where(input_df['codes']=='I5040','1',np.where(input_df['codes']=='I5041','1',np.where(input_df['codes']=='I5042','1',np.where(input_df['codes']=='I5043','1',np.where(input_df['codes']=='I50810','1',np.where(input_df['codes']=='I50811','1',np.where(input_df['codes']=='I50812','1',np.where(input_df['codes']=='I50813','1',np.where(input_df['codes']=='I50814','1',np.where(input_df['codes']=='I5082','1',np.where(input_df['codes']=='I5083','1',np.where(input_df['codes']=='I5084','1',np.where(input_df['codes']=='I5089','1',np.where(input_df['codes']=='I509','1','0')))))))))))))))))))))))))))))


input_df['here_flag_incl_e851'] = np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E851','1',np.where(input_df['codes']=='E852','1','0')))

input_df
all_pat = input_df[['patient_id','WT_flag','hf_flag','here_flag_incl_e851']]
wt_temp = input_df[input_df['WT_flag']=='1']

wt_count = wt_temp.patient_id.nunique()


here_temp=input_df[input_df['here_flag_incl_e851']=='1']
here_count = here_temp.patient_id.nunique()

wt_temp=wt_temp.append(here_temp,ignore_index=True)
wt_count = wt_temp.patient_id.nunique()


wt_pat = pd.DataFrame(wt_temp.patient_id.unique(),columns = ['patient_id'])
wt_pat['ptid'] = wt_pat['patient_id']
hf_temp = input_df[input_df['hf_flag']=='1']
hf_pat = pd.DataFrame(hf_temp.patient_id.unique(),columns = ['patient_id'])
final_wt_pat = wt_pat.merge(hf_pat, on = 'patient_id', how = 'inner')
final_hf_pat = hf_pat.merge(wt_pat,on='patient_id',how = 'left')
final_hf_pat['check'] = np.where(final_hf_pat['patient_id']==final_hf_pat['ptid'],'1','0')
final_hf_pat = final_hf_pat[final_hf_pat['check']=='0']
total_wt_pt_count = final_wt_pat.shape[0]
total_hf_pt_count = final_hf_pat.shape[0]
df_1_1 = pd.DataFrame()
df_1_9 = pd.DataFrame()
df_1_19 = pd.DataFrame()


if total_wt_pt_count > total_hf_pt_count:
    print("Not Enough Heart Failure Patients To Create Prediction Cohorts")
if 19*total_wt_pt_count<=total_hf_pt_count:
    print("Creating 1:19 cohort...")
    df_1_19 = pd.concat([final_wt_pat,final_hf_pat.sample(21*total_wt_pt_count,random_state=1337,replace=False)])
    df_1_19 = pd.DataFrame(df_1_19['patient_id'],columns = ['patient_id'])
elif 9*total_wt_pt_count<=total_hf_pt_count:
    print("Creating 1:9 cohort...")
    df_1_9 = pd.concat([final_wt_pat,final_hf_pat.sample(11*total_wt_pt_count,random_state=1337,replace=False)])
    df_1_9 = pd.DataFrame(df_1_9['patient_id'],columns = ['patient_id'])
elif total_wt_pt_count<=total_hf_pt_count:
    print("Creating 1:1 cohort...")
    df_1_1 = pd.concat([final_wt_pat,final_hf_pat.sample(2*total_wt_pt_count,random_state=1337,replace=False)])
    df_1_1 = pd.DataFrame(df_1_1['patient_id'],columns = ['patient_id'])
            
    
run_wt_1_1 = not df_1_1.empty
run_wt_1_9 =  not df_1_9.empty
run_wt_1_19 =  not df_1_19.empty

os.chdir(input_dir)
    
if run_wt_1_19:
    cleaned_input = df_1_19.merge(converted_df,on = 'patient_id', how = 'inner')
    print("Cohort Creation Sucessful. Please look for cleaned_input.csv in the same folder")
    cleaned_input = cleaned_input[['patient_id','codes']]
    cleaned_input['type'] = 2
    cleaned_input.to_csv('cleaned_input.csv')
elif run_wt_1_9: 
    cleaned_input = df_1_9.merge(converted_df,on = 'patient_id', how = 'inner')
    print("Cohort Creation Sucessful. Please look for cleaned_input.csv in the same folder")
    cleaned_input = cleaned_input[['patient_id','codes']]
    cleaned_input['type'] = 2
    cleaned_input.to_csv('cleaned_input.csv')
elif run_wt_1_1:
    cleaned_input = df_1_1.merge(converted_df,on = 'patient_id', how = 'inner')
    print("Cohort Creation Sucessful. Please look for cleaned_input.csv in the same folder")
    cleaned_input = cleaned_input[['patient_id','codes']]
    cleaned_input['type'] = 2
    cleaned_input.to_csv('cleaned_input.csv')
else:
    print("Not Enough Patients for cohort Creation")

