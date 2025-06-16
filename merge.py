import pandas as pd

# 1. Load
fake_df = pd.read_csv('fake.csv')
real_df = pd.read_csv('true.csv')

# 2. Label them
fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

# 3. Merge title + text (if both exist)
if 'title' in fake_df.columns and 'text' in fake_df.columns:
    fake_df['text'] = fake_df['title'] + " " + fake_df['text']
    real_df['text'] = real_df['title'] + " " + real_df['text']

# 4. Keep only needed columns
fake_df = fake_df[['text', 'label']]
real_df = real_df[['text', 'label']]

# 5. Merge, shuffle, limit to 2000 rows
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[:2000]  # lightweight!

# 6. Save
df.to_csv('dataset.csv', index=False)
print("dataset.csv created!")
