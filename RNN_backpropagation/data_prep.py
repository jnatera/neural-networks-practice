import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
print('Data en bruto')
print(admissions)
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
print("data Dimensionada y Concatenada")
print(data.head())
data = data.drop('rank', axis=1)
print("data Estandarizada")
# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    print(field + " mean:" + str(mean) + ' std: ' + str(std))
    data.loc[:,field] = (data[field]-mean)/std
print(data.head()) 

# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
print("\nSamples de indices para recortar y tomar data de entrenamiento y data test")
print(sample[:10])
data, test_data = data.iloc[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
print("\nFeatures o entradas para entrenar")
print(features.head())
print("\nTargets o valores esperados") 
print(targets.head())
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
#quit()