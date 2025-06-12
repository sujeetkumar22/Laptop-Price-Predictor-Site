import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load and preprocess data
df = pd.read_csv('laptop_data.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Process RAM and Weight
df['Ram'] = df['Ram'].str.replace('GB','').astype('int32')
df['Weight'] = df['Weight'].str.replace('kg','').astype('float32')

# Process screen resolution and create features
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Extract resolution
def get_resolution(x):
    for part in x.split():
        if 'x' in part:
            return part
    return None

df['resolution'] = df['ScreenResolution'].apply(get_resolution)
df[['X_res', 'Y_res']] = df['resolution'].str.split('x', expand=True).astype('int')
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

# Process CPU
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

# Process GPU
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])
df = df[df['Gpu brand'] != 'ARM']

# Process Operating System
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
df['os'] = df['OpSys'].apply(cat_os)

# Process storage (HDD/SSD)
def get_storage(text, storage_type):
    text = str(text).upper()
    if storage_type.upper() in text:
        parts = text.split('+')
        for part in parts:
            if storage_type.upper() in part:
                value = ''.join(filter(str.isdigit, part))
                if value:
                    if 'TB' in part:
                        return int(value) * 1000
                    return int(value)
    return 0

df['Memory'] = df['Memory'].astype(str)
df['HDD'] = df['Memory'].apply(lambda x: get_storage(x, 'HDD'))
df['SSD'] = df['Memory'].apply(lambda x: get_storage(x, 'SSD'))

# Select features for model
X = df[['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 
        'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']]
y = np.log(df['Price'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Create pipeline
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0,1,7,10,11])
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                            random_state=3,
                            max_samples=0.5,
                            max_features=0.75,
                            max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Train the model
pipe.fit(X_train, y_train)

# Save the model and dataframe
pickle.dump(df, open('df.pkl','wb'))
pickle.dump(pipe, open('pipe.pkl','wb'))

print("Model retrained and saved successfully!")
