import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Load the dataset
    df = pd.read_csv('survey.csv')
    
    # Handle missing values
    df = df.fillna('Missing')
    
    # Set the style for better visualizations
    plt.style.use('seaborn-v0_8')
    
    # Create a figure with subplots
    plt.figure(figsize=(20, 15))
    
    # 1. Treatment Distribution
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='treatment')
    plt.title('Distribution of Mental Health Treatment')
    plt.xticks(rotation=45)
    
    # 2. Age Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='Age', bins=30)
    plt.title('Age Distribution')
    
    # 3. Gender Distribution
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='Gender')
    plt.title('Gender Distribution')
    plt.xticks(rotation=45)
    
    # 4. Work Interference Distribution
    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='work_interfere')
    plt.title('Work Interference Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribution_plots.png')
    plt.close()
    
    # Create another figure for relationship plots
    plt.figure(figsize=(20, 15))
    
    # 1. Treatment by Remote Work
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='remote_work', hue='treatment')
    plt.title('Treatment by Remote Work Status')
    plt.xticks(rotation=45)
    
    # 2. Treatment by Company Size
    plt.subplot(2, 2, 2)
    sns.countplot(data=df, x='no_employees', hue='treatment')
    plt.title('Treatment by Company Size')
    plt.xticks(rotation=45)
    
    # 3. Treatment by Tech Company
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='tech_company', hue='treatment')
    plt.title('Treatment by Tech Company Status')
    plt.xticks(rotation=45)
    
    # 4. Treatment by Family History
    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='family_history', hue='treatment')
    plt.title('Treatment by Family History')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('relationship_plots.png')
    plt.close()
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print("\nTotal number of responses:", len(df))
    print("\nTreatment distribution:")
    print(df['treatment'].value_counts(normalize=True) * 100)
    print("\nMissing values:")
    print(df.isnull().sum())

except FileNotFoundError:
    print("Error: survey.csv file not found. Please make sure the file exists in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")