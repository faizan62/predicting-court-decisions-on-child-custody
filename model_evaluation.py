import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

def load_model(model_path):
    # Assuming the model is a pickled sklearn model, modify as necessary
    import joblib
    return joblib.load(model_path)

def load_test_data(file_path):
    return pd.read_csv(file_path)

def prepare_features(df):
    # Assuming the features are prepared as in the training phase
    df['Plaintiff\'s Custody Request'] = df['Plaintiff\'s Custody Request'].astype('category').cat.codes
    df['Court\'s Ruling'] = df['Court\'s Ruling'].astype('category').cat.codes
    X = df.drop('Court\'s Ruling', axis=1)
    y = df['Court\'s Ruling']
    return X, y

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))
    print("Accuracy Score:", accuracy_score(y, predictions))
    print("F1 Score:", f1_score(y, predictions, average='weighted'))
    # Uncomment the next line if your model probability estimates are applicable
    # print("ROC AUC Score:", roc_auc_score(y, model.predict_proba(X)[:, 1]))

def main():
    model_path = 'final_model.pkl'
    test_data_path = 'Cleaned_Test_Dataset.csv'
    
    model = load_model(model_path)
    test_data = load_test_data(test_data_path)
    X_test, y_test = prepare_features(test_data)
    
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
