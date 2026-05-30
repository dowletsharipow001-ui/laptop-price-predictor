import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from pathlib import Path

current_file = Path(__file__).resolve()
csv_path = current_file.parents[1] / 'data' / 'raw' / 'laptop_price.csv'

df = pd.read_csv(csv_path, encoding='latin1')

# Информация про данных
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print("---Информация---")
print(df.head())
print(df.info())
print(df.describe())


df_2 = df.copy()

# Убирание строки в признаках Weight, Ram
df_2["Weight"] = df_2["Weight"].str.replace('kg', '', regex=False).str.strip().astype(float)
df_2["Ram"] = df_2["Ram"].str.replace('GB', '', regex=False).str.strip().astype(float)

# ScreenResolution
screen_models = ["Full HD", "IPS Panel","Retina Display","Touchscreen","Quad HD+","4K Ultra HD"]
for el in screen_models:
    df_2[f'is_{el}'] = df_2["ScreenResolution"].str.contains(f'{el}', regex=False, na=False).astype(int)
extracted = df_2['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(int)
df_2['ScreenSize'] = extracted[0] * extracted[1]

# Memory
def extract_memory_features(memory_str):
    ssd = hdd = flash = hybrid = 0
    
    memory_str = str(memory_str).strip()
    
    pattern = r'(\d+(?:\.\d+)?)\s*(GB|TB)\s*(SSD|HDD|Flash Storage|Hybrid)'
    matches = re.findall(pattern, memory_str)
    
    for amount, unit, mem_type in matches:
        amount = float(amount)
        
        if unit == 'TB':
            amount *= 1024


        if mem_type == 'SSD':
            ssd += amount
        elif mem_type == 'HDD':
            hdd += amount
        elif mem_type == 'Flash Storage':
            flash += amount
        elif mem_type == 'Hybrid':
            hybrid += amount
            
    return pd.Series([int(ssd), int(hdd), int(flash), int(hybrid)])

df_2[['ssd_gb', 'hdd_gb', 'flash_gb', 'hybrid_gb']] = df_2['Memory'].apply(extract_memory_features)

# CPU и GPU
df_2['Cpu_Company'] = df_2['Cpu'].str.split().str[0]
df_2['Cpu_Frequency_GHz'] = df_2['Cpu'].str.extract(r'(\d+\.?\d*)\s*GHz').astype(float)

cpu_pattern = r'(i7|i5|i3|Celeron|Pentium|Ryzen|A-Series|A10|A8|A6|A9)'
df_2['Cpu_Type'] = df_2['Cpu_Company'] + ' ' + df_2['Cpu'].str.extract(cpu_pattern)[0].fillna('Other')
df_2['Cpu_Type'] = df_2['Cpu_Type'].str.replace(r'AMD (A10|A8|A6|A9)', 'AMD A-Series', regex=True)

df_2['Gpu_Company'] = df_2['Gpu'].str.split().str[0]
df_2['is_Intel_Gpu'] = (df_2['Gpu_Company'] == 'Intel').astype(int)

gpu_map = r'(GeForce|Quadro|Radeon)'
df_2['Gpu_Type'] = df_2['Gpu_Company'] + ' ' + df_2['Gpu'].str.extract(gpu_map)[0].fillna('Graphics')

df_2 = df_2.drop(columns=['Cpu', 'Gpu', "Memory", "ScreenResolution"])




plt.figure(figsize=(12, 8))
sns.histplot(data=df_2, x="Price_euros", bins=30, kde=True)
plt.show()

df_2["Price_euros"] = np.log1p(df_2["Price_euros"])

plt.figure(figsize=(12, 8))
sns.histplot(data=df_2, x="Price_euros", bins=30, kde=True)
plt.show()

processed_csv_path = current_file.parents[1] / 'data' / 'processed' / 'laptop_price_cleaned.csv'

df_2.to_csv(processed_csv_path, index=False)