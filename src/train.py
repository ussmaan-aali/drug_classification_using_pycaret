# Importing necessary libraries
import pandas as pd
from pycaret.classification import *
  
# Reading the data
data = pd.read_csv('data\drug_classification.csv')

# Initializing classification setup
clf = setup(data=data, target='Drug', train_size=0.86, session_id=123, use_gpu = True, experiment_name='classifying_drugs')

# Comparing models
best_model = compare_models(sort='F1', n_select=1)

# predict on test set
holdout_pred = predict_model(best_model)

# Evaluating the best model
# evaluate_model(best_model)

# Saving the best model
save_model(best_model, 'classification')