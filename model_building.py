import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv('laptop_data.csv')

# Preprocessing
df['Ram'] = df['Ram'].str.replace('GB','').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg','').astype('float32')
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

def fetch_processor(text):
    if 'Intel Core i7' in text: return 'i7'
    if 'Intel Core i5' in text: return 'i5'
    if 'Intel Core i3' in text: return 'i3'
    elif 'AMD' in text: return 'AMD'
    else: return 'Other'
df['Cpu Brand'] = df['Cpu'].apply(fetch_processor)

# Define the exact features we want to use
X = df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'OpSys', 'Cpu Brand', 'Inches']]
y = np.log(df['Price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Column Transformer using named columns
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), 
     ['Company', 'TypeName', 'OpSys', 'Cpu Brand'])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100, random_state=3)

pipe = Pipeline([('step1', step1), ('step2', step2)])
pipe.fit(X_train, y_train)

# Save
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Model trained successfully with column names!")