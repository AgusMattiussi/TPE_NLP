from collections import defaultdict
import numpy as np
import pandas as pd

def count_word_amount_per_article(articles: list):
    label_data = defaultdict(int)

    for article in articles:
        words = article.split()
        for word in words:
            label_data[word] += 1
    return label_data

def get_label_tf(label: str, articles: list):
    label_data = count_word_amount_per_article(articles)

    word_count = np.size(list(label_data.keys()))
    tf_words={}

    for word, count in label_data.items():
        tf_words[word] = count/word_count

if __name__ == '__main__':
    df = pd.read_csv('../data/bbc_data.csv')
    labeled_articles = defaultdict(list)
    
    for _, row in df.iterrows():
        labeled_articles[row['labels']].append(row['data'])

    for label, articles in labeled_articles.items():
        get_label_tf(label, articles)