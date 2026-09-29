import pandas as pd
import numpy as np

# 1. Clean Dataset (scores 90+)
print("Creating clean dataset...")
clean_data = {
    'age': np.random.randint(20, 65, 1000),
    'income': np.random.randint(30000, 150000, 1000),
    'tenure': np.random.randint(1, 30, 1000),
    'purchases': np.random.randint(5, 100, 1000),
    'satisfaction': np.random.randint(1, 10, 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.85, 0.15])
}
pd.DataFrame(clean_data).to_csv('data/demo_clean.csv', index=False)

# 2. Problematic Dataset (scores 60-70)
print("Creating problematic dataset...")
problem_data = clean_data.copy()
problem_df = pd.DataFrame(problem_data)
# Add missing values
problem_df.loc[np.random.choice(1000, 180, replace=False), 'income'] = np.nan
# Add duplicates
problem_df = pd.concat([problem_df, problem_df.sample(35)])
# Severe imbalance
problem_df['target'] = np.random.choice([0, 1], len(problem_df), p=[0.95, 0.05])
problem_df.to_csv('data/demo_problematic.csv', index=False)

# 3. Terrible Dataset (scores <40)
print("Creating terrible dataset...")
terrible_data = clean_data.copy()
terrible_df = pd.DataFrame(terrible_data)
# Lots of missing
terrible_df.loc[np.random.choice(1000, 450, replace=False), 'income'] = np.nan
terrible_df.loc[np.random.choice(1000, 350, replace=False), 'age'] = np.nan
# Many duplicates
terrible_df = pd.concat([terrible_df, terrible_df.sample(200)])
# Extreme imbalance
terrible_df['target'] = np.random.choice([0, 1], len(terrible_df), p=[0.98, 0.02])
terrible_df.to_csv('data/demo_terrible.csv', index=False)

print("\n✅ Demo datasets created!")
print("   - data/demo_clean.csv")
print("   - data/demo_problematic.csv")
print("   - data/demo_terrible.csv")