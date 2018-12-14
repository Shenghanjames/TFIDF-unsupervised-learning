"""
Goal: data_normalization is the package about helping doing preprocessing of the data from dag_food and laptops
Author: Shenghan Zhang
coding:utf-8
date: 5th, Dec, 2018
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class TFIDFConstructor:
    def __init__(self,corpus):
        # here we should input the data with the structure: the list of all data, each element of the list should be string
        self.corpus=corpus

    def convert_tfidf_matrix(self):
        vectorizer=CountVectorizer()
        X=vectorizer.fit_transform(self.corpus)
        unique_tokens=vectorizer.get_feature_names()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X)
        return unique_tokens,tfidf

