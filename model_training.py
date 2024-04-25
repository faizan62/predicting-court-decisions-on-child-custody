import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Example preprocessing steps
    df = df.dropna(subset=['Plaintiff\'s Custody Request', 'Court\'s Ruling'])
    # Convert categories to numeric labels
    df['Plaintiff\'s Custody Request'] = df['Plaintiff\'s Custody Request'].astype('category').cat.codes
    df['Court\'s Ruling'] = df['Court\'s Ruling'].astype('category').cat.codes
    return df

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

def main():
    data_path = 'Cleaned_Training_Dataset.csv'
    df = load_data(data_path)
    df_preprocessed = preprocess_data(df)
    
    X = df_preprocessed.drop('Court\'s Ruling', axis=1)
    y = df_preprocessed['Court\'s Ruling']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
