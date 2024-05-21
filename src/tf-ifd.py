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
    # Train
    train_df = pd.DataFrame({'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy'], 'Value': [f1_score(y_train, model.predict(X_train), average='macro'), precision_score(y_train, model.predict(X_train), average='macro'), recall_score(y_train, model.predict(X_train), average='macro'), accuracy_score(y_train, model.predict(X_train))]})
    train_df['Model'] = 'Train'

    # Test   
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    test_df = pd.DataFrame({'Metric': ['F1 Score', 'Precision', 'Recall', 'Accuracy'], 'Value': [f1, precision, recall, accuracy]})
    test_df['Model'] = 'Test'

    # Combinar los DataFrames
    combined_df = pd.concat([train_df, test_df])
    pivot_df = combined_df.pivot(index='Metric', columns='Model', values='Value')
    return pivot_df

if __name__ == '__main__':
    df = pd.read_csv('../data/bbc_data.csv')
    labeled_articles = defaultdict(list)
    labels_idx = {label: idx for idx, label in enumerate(df['labels'].unique())}
    y = []

    for _, row in df.iterrows():
        y.append(labels_idx[row['labels']])

    vectorizer = TfidfVectorizer(stop_words='english')
    # This does:
    # tf_word = word_appearances_amount_per_article/total_amount_of_words_per_article
    # idf_word = math.log10(amount_of_articles / (float(word_appearances_in_all_articles) + 1))
    # tf_idf_word = tf_word * idf_word
    X = vectorizer.fit_transform(df['data'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model and evaluate
    model = train(X_train, y_train)
    metric_df = evaluate_model(model, X_train, X_test, y_train, y_test)