from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def train(X_train, y_train, max_iter=10000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    train_df = pd.DataFrame({'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy'], 'Value': [f1_score(y_train, model.predict(X_train), average='macro'), precision_score(y_train, model.predict(X_train), average='macro'), recall_score(y_train, model.predict(X_train), average='macro'), accuracy_score(y_train, model.predict(X_train))]})
    test_df = pd.DataFrame({'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy'], 'Value': [f1, precision, recall, accuracy]})

    train_df['Model'] = 'Train'
    test_df['Model'] = 'Test'

    # Combinar los DataFrames
    combined_df = pd.concat([train_df, test_df])
    pivot_df = combined_df.pivot(index='Metric', columns='Model', values='Value')
    print(pivot_df)

if __name__ == '__main__':
    df = pd.read_csv('../data/bbc_data.csv')
    labeled_articles = defaultdict(list)
    english_stopwords = read_english_stopwords("../data/english.txt")

    labels_idx = {label: idx for idx, label in enumerate(df['labels'].unique())}
    X = []
    y = []

    for _, row in df.iterrows():
        y.append(labels_idx[row['labels']])

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    # Fit and transform the data to create the TF-IDF matrix
    X = vectorizer.fit_transform(df['data'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define model and evaluate
    model = train(X_train, y_train)

    accuracy, recall, precision, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)