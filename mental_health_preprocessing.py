import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load raw data
df = pd.read_csv('survey.csv')

def clean_gender(gender):
    if pd.isna(gender):
        return 'Unknown'
    gender = str(gender).lower().strip()
    if gender in ['m', 'male', 'male-ish', 'maile', 'mal', 'male (cis)', 'make', 'male ', 'man', 'msle', 'mail', 'malr', 'cis male']:
        return 'Male'
    elif gender in ['f', 'female', 'woman', 'femake', 'female ', 'cis female', 'cis-female/femme', 'female (cis)', 'femail']:
        return 'Female'
    else:
        return 'Other'

def preprocess_data(df):
    df_cleaned = df.copy()
    
    # Drop columns that hurt accuracy or add noise
    drop_cols = ['Timestamp', 'comments', 'state', 'Country']
    df_cleaned.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Clean gender
    df_cleaned['Gender'] = df_cleaned['Gender'].apply(clean_gender)
    
    # Clean age
    df_cleaned['Age'] = pd.to_numeric(df_cleaned['Age'], errors='coerce')
    df_cleaned = df_cleaned[(df_cleaned['Age'] >= 18) & (df_cleaned['Age'] <= 100)]
    
    # Fill missing values
    df_cleaned['self_employed'] = df_cleaned['self_employed'].fillna('No')
    df_cleaned['work_interfere'] = df_cleaned['work_interfere'].fillna("Don't know")
    
    # List of categorical columns after dropping 'Country'
    categorical_columns = ['Gender', 'self_employed', 'family_history', 
                          'treatment', 'work_interfere', 'no_employees', 'remote_work',
                          'tech_company', 'benefits', 'care_options', 'wellness_program',
                          'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                          'phys_health_consequence', 'coworkers', 'supervisor',
                          'mental_health_interview', 'phys_health_interview',
                          'mental_vs_physical', 'obs_consequence']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))
        label_encoders[col] = le
    
    return df_cleaned, label_encoders

if __name__ == '__main__':
    df_cleaned, label_encoders = preprocess_data(df)
    df_cleaned.to_csv('preprocessed_survey.csv', index=False)
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("Preprocessing complete. Saved preprocessed_survey.csv and label_encoders.pkl")
