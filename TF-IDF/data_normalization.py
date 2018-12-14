"""
Effect: data_normalization is the package about helping doing preprocessing of the data from dag_food and laptops
Author: Shenghan Zhang
coding:utf-8
date: 5th,Dec, 2018

"""
import nltk
# because I didn't install the dataset of nltk with its default path, I add the installed path for nltk to work
nltk.data.path.append("D:\\workspace\\nltk_package")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class Preprocessor:
    def __init__(self,data):
        # the data must be string
        self.data=data

    def tokenization(self):
        """
        here to do the tokenization of the string, classify the string into tokens

        """
        tokenized_data=word_tokenize(self.data)
        return tokenized_data

    def stopwords_removal(self):
        """
        here is to remove the stopwords

        """
        tokenized_data=self.tokenization()
        stop_words = set(stopwords.words('english'))
        filtered_words = []
        for w in tokenized_data:
            if w not in stop_words:
                filtered_words.append(w)
        return filtered_words

    def stemming(self):
        """
        here is to stem the core part of each word
        """
        filtered_words=self.stopwords_removal()
        ps = PorterStemmer()
        stemming_words=list()
        for w in filtered_words:
            stemming_words.append(ps.stem(w))
        return stemming_words

    def get_string(self):
        """
        here to change the tokenized list to string again, in order to make sure that skicit-learn can be used

        """
        stemming_words=self.stemming()
        str_sentence = ' '.join(str(e) for e in stemming_words)
        return str_sentence

class Series_Preprocessor():
    """
    this class is built for dealing with the series of data
    each element of data is string type
    """
    def __init__(self,list_data):
        self.list_data=list_data
        self.length=len(list_data)

    def get_data(self):
        """
        to get normalized tokens of each description or name with the type of string:
        """
        normalized_list_data=list()
        for no in range(self.length):
            self.data=self.list_data[no]
            self.data=str(self.data)
            pp=Preprocessor(self.data)
            normalized_data=pp.get_string()
            normalized_list_data.append(normalized_data)

        return normalized_list_data

