import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the preprocessed data
df = pd.read_csv('preprocessed_survey.csv')

# Set up the plotting style
plt.style.use('seaborn-v0_8')

# Create a figure for correlation matrix
plt.figure(figsize=(15, 12))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Create subplots for feature distributions by treatment
features = ['Age', 'Gender', 'family_history', 'work_interfere', 'no_employees', 
            'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 4, i)
    if df[feature].dtype in ['int64', 'float64']:
        sns.boxplot(x='treatment', y=feature, data=df)
    else:
        sns.countplot(data=df, x=feature, hue='treatment')
    plt.title(f'{feature} by Treatment')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Create pairplot for numerical features
numerical_features = ['Age'] + [col for col in df.columns if df[col].dtype in ['int64', 'float64']][:5]
plt.figure(figsize=(15, 15))
sns.pairplot(df[numerical_features + ['treatment']], hue='treatment')
plt.savefig('feature_pairplot.png')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(df.describe())

print("\nClass Distribution:")
print(df['treatment'].value_counts(normalize=True) * 100)