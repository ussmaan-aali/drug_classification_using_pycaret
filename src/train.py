# pip install pycaret
# pip install --user mlflow

# Importing necessary libraries
import pandas as pd
from pycaret.classification import *

import pandas as pd
  
# Reading the data
data = pd.read_csv('/content/drug_classification.csv')



# Initializing classification setup
clf = setup(data=data, target='Drug', train_size=0.86, session_id=123, use_gpu = True, experiment_name='classifying_drugs')

print(clf)

# Comparing models
best_model = compare_models(sort='F1', n_select=1)

# Evaluating the best model
evaluate_model(best_model)



# Saving the best model
save_model(best_model, 'classification')