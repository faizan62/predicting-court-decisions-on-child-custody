import pandas as pd

def load_data(file_path):
    # Load data from a CSV file
    return pd.read_csv(file_path)

def clean_data(df):
    # Remove any rows with missing values in essential columns
    df = df.dropna(subset=['Plaintiff\'s Custody Request', 'Court\'s Ruling', 'Pivotal Arguments'])
    # Normalize text data, e.g., converting to lowercase
    df['Plaintiff\'s Custody Request'] = df['Plaintiff\'s Custody Request'].str.lower()
    df['Court\'s Ruling'] = df['Court\'s Ruling'].str.lower()
    df['Pivotal Arguments'] = df['Pivotal Arguments'].str.lower()
    return df

def save_data(df, output_path):
    # Save the cleaned data back to a new CSV file
    df.to_csv(output_path, index=False)
    print('Cleaned data saved to:', output_path)

def main():
    input_path = 'Training_Dataset.csv'
    output_path = 'Cleaned_Training_Dataset.csv'
    
    # Load, clean, and save the dataset
    data = load_data(input_path)
    cleaned_data = clean_data(data)
    save_data(cleaned_data, output_path)

if __name__ == '__main__':
    main()
