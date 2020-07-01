
## Importing libraries

import pandas as pd
import numpy as np
import sklearn.metrics as metrics


## Creating flags for amyloidosis codes at patient level
class Flags:
	def __init__(self,input_df):
		self.amyloidosis_flag = input_df
		self.amyloidosis_flag['E85_flag'] = np.where(input_df['codes']=='E85','1','0')
		self.amyloidosis_flag['E850_flag'] = np.where(input_df['codes']=='E850','1','0')
		self.amyloidosis_flag['E851_flag'] = np.where(input_df['codes']=='E851','1','0')
		self.amyloidosis_flag['E852_flag'] = np.where(input_df['codes']=='E852','1','0')
		self.amyloidosis_flag['E853_flag'] = np.where(input_df['codes']=='E853','1','0')
		self.amyloidosis_flag['E854_flag'] = np.where(input_df['codes']=='E854','1','0')
		self.amyloidosis_flag['E858_flag'] = np.where(input_df['codes']=='E858','1','0')
		self.amyloidosis_flag['E8581_flag'] = np.where(input_df['codes']=='E8581','1','0')
		self.amyloidosis_flag['E8582_flag'] = np.where(input_df['codes']=='E8582','1','0')
		self.amyloidosis_flag['E8589_flag'] = np.where(input_df['codes']=='E8589','1','0')
		self.amyloidosis_flag['E859_flag'] = np.where(input_df['codes']=='E859','1','0')
		self.amyloidosis_flag['hereditary_flag'] = np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E852','1','0'))
		self.amyloidosis_flag['hereditary_incl_E851_flag'] = np.where(input_df['codes']=='E850','1',np.where(input_df['codes']=='E851','1',np.where(input_df['codes']=='E852','1','0')))
		self.amy_E85 = pd.DataFrame({'max_E85_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E85_flag']}).reset_index()
		self.amy_E850 = pd.DataFrame({'max_E850_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E850_flag']}).reset_index()
		self.amy_E851 = pd.DataFrame({'max_E851_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E851_flag']}).reset_index()
		self.amy_E852 = pd.DataFrame({'max_E852_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E852_flag']}).reset_index()
		self.amy_E853 = pd.DataFrame({'max_E853_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E853_flag']}).reset_index()
		self.amy_E854 = pd.DataFrame({'max_E854_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E854_flag']}).reset_index()
		self.amy_E858 = pd.DataFrame({'max_E858_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E858_flag']}).reset_index()
		self.amy_E8581 = pd.DataFrame({'max_E8581_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E8581_flag']}).reset_index()
		self.amy_E8582 = pd.DataFrame({'max_E8582_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E8582_flag']}).reset_index()
		self.amy_E8589 = pd.DataFrame({'max_E8589_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E8589_flag']}).reset_index()
		self.amy_E859 = pd.DataFrame({'max_E859_flag' : self.amyloidosis_flag.groupby('patient_id').max()['E859_flag']}).reset_index()
		self.amy_hereditary = pd.DataFrame({'max_hereditary_flag' : self.amyloidosis_flag.groupby('patient_id').max()['hereditary_flag']}).reset_index()
		self.amy_hereditary = pd.DataFrame({'max_hereditary_incl_E851_flag' : self.amyloidosis_flag.groupby('patient_id').max()['hereditary_incl_E851_flag']}).reset_index()
		

target_map = {u'1': 1, u'0': 0}	

## Predicting patients using trained model. Probabilities and predictions are included in the output file.

def predictions(model_file,test_df):

	rf_random_predictions = model_file.predict(test_df)
	rf_random_probas = model_file.predict_proba(test_df)
	rf_random_predictions = pd.Series(data=rf_random_predictions, index=test_df.index, name='predicted_value')
	cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
	]
	rf_random_probabilities = pd.DataFrame(data=rf_random_probas, index=test_df.index, columns=cols)
	results_test_rf_random = pd.concat([rf_random_predictions,rf_random_probabilities],axis=1)
	results_test_rf_random.reset_index(level=0, inplace=True)
	return results_test_rf_random

## Class to calculated model metrics.
	
class custom_model_metrics:
		def __init__(self,final,y_test):
			self.preds = final['probability_of_value_1']
			self.fpr, self.tpr, self.threshold = metrics.roc_curve(y_test, self.preds)
			self.roc_auc = metrics.auc(self.fpr, self.tpr)
			self.confusion_matrix = metrics.confusion_matrix(y_test, final['predicted_value'])
			
			## Calling grid of confusion matrix
			self.tp = self.confusion_matrix[1][1]
			self.tn = self.confusion_matrix[0][0]
			self.fp = self.confusion_matrix[0][1]
			self.fn = self.confusion_matrix[1][0]
			
			
			try:
				## Calculating specificity
				self.specificity= self.tn/(self.tn+self.fp)
			
				## Calculating PPV
				self.ppv= self.tp/(self.tp+self.fp)
			
				## Calculating NPV
				self.npv=self.tn/(self.tn+self.fn)
							
			except:
				print("Denominator became 0") 
				exit(0) 


			## Calculating recall
			self.recall = metrics.recall_score(y_test, final['predicted_value'])
			
			
			## Calculating accuracy
			self.accuracy = metrics.accuracy_score(y_test, final['predicted_value'])
			
			## Calculating F1 Score
			self.f1_score = metrics.f1_score(y_test, final['predicted_value'])
			
			##self.recall = metrics.recall_score(y_test, final['predicted_value'])
			##self.precision = metrics.precision_score(y_test, final['predicted_value'])
			
			self.classification_report = metrics.classification_report(y_test, final['predicted_value'])
			self.output_matrix = pd.DataFrame()
			
			## Storing metrics in output file 
			
			self.output_matrix.loc[0,1]='Sensitivity:'
			self.output_matrix.loc[0,2]= self.recall
			self.output_matrix.loc[1,1]='Specificity'
			self.output_matrix.loc[1,2]= self.specificity
			self.output_matrix.loc[2,1]='PPV'
			self.output_matrix.loc[2,2]= self.ppv
			self.output_matrix.loc[3,1]='NPV'
			self.output_matrix.loc[3,2]= self.npv
			self.output_matrix.loc[4,1]='Accuracy'
			self.output_matrix.loc[4,2]= self.accuracy
			self.output_matrix.loc[5,1]= 'ROC'
			self.output_matrix.loc[5,2]= self.roc_auc
						
			
			## Storing confusion metrics in output file 
			
			self.output_matrix.columns=[['metric','value']]
			self.output_matrix = self.output_matrix.reset_index()
			self.output_matrix_2 = pd.DataFrame()
			self.output_matrix_2.loc[0,0] = "Confusion"
			self.output_matrix_2.loc[0,1] = "Matrix"
			self.output_matrix_2.loc[0,2] = "Predicted"
			self.output_matrix_2.loc[0,3] = "Predicted"
			self.output_matrix_2.loc[1,2] = 1
			self.output_matrix_2.loc[1,3] = 0
			self.output_matrix_2.loc[2,0] = "Actual"
			self.output_matrix_2.loc[2,1] = 1
			self.output_matrix_2.loc[3,0] = "Actual"
			self.output_matrix_2.loc[3,1] = 0
			self.cols=[0,1,2,3]
			self.output_matrix_2 = self.output_matrix_2[self.cols]
			self.output_matrix_2.loc[2,2] = self.tp
			self.output_matrix_2.loc[3,3] = self.tn
			self.output_matrix_2.loc[3,2] = self.fp
			self.output_matrix_2.loc[2,3] = self.fn
			self.output_matrix_2.loc[4,0] = ""
			self.output_matrix_2.loc[4,1] = ""
			self.output_matrix_2.loc[4,2] = ""
			self.output_matrix_2.loc[4,3] = ""
			self.output_matrix_2 = self.output_matrix_2[self.cols]



	