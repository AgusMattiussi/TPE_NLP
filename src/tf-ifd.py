from collections import defaultdict
import math
import re
import string
import numpy as np
import pandas as pd

def read_english_stopwords(file_path):
    word_set = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            word_set.update(words)
    return word_set


def count_word_amount_per_article(articles: list):
    label_data = defaultdict(int)

    for article in articles:
        words = article.split()
        for word in words:
            label_data[word] += 1
    return label_data

def get_label_tf(label_data: defaultdict):
    word_count = np.size(list(label_data.keys()))
    tf_words={}

    for word, count in label_data.items():
        tf_words[word] = count/word_count

    return tf_words

def get_label_idf(label_tf: dict):
    N = len(label_tf)
    idf_words = {}
    
    idf_words = dict.fromkeys(label_tf.keys(), 0)
    for word, val in idf_words.items():
        idf_words[word] = math.log10(N / (float(val) + 1))
    
    return idf_words

if __name__ == '__main__':
    df = pd.read_csv('../data/bbc_data.csv')
    labeled_articles = defaultdict(list)
    english_stopwords = read_english_stopwords("../data/english.txt")
    
    for _, row in df.iterrows():
        words = row['data'].split()
        
        filtered_words = []
        for word in words:
            word = word.strip(string.punctuation)
            if word.lower() not in english_stopwords:
                filtered_words.append(word)
        cleaned_data = ' '.join(filtered_words)
        
        labeled_articles[row['labels']].append(cleaned_data)


    for label, articles in labeled_articles.items():
        label_data = count_word_amount_per_article(articles)
        label_tf = get_label_tf(label_data)
        label_idf = get_label_idf(label_tf)