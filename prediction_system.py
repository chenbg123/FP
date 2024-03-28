# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

import re
import string
import nltk
import pickle
import streamlit as st

# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

loaded_model_LR=pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/trained_model_LR.sav",'rb'))
loaded_model_DT=pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/trained_model_DT.sav",'rb'))
loaded_model_RFC=pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/trained_model_RFC.sav",'rb'))
loaded_model_GBC=pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/trained_model_GBC.sav",'rb'))
loaded_model_voting=pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/trained_model_voting.sav",'rb'))
loaded_vectorization = pickle.load(open("D:/kylec/Documents/Programming Files/project/Text-analytics-on-deceptive-and-fake-information-on-COVID-19-in-Malaysia/Fake_news/model/vectorization.pickle", "rb"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) #\[: 匹配左方括号，\是转义字符，因为方括号在正则表达式中是特殊字符，需要转义。
                                       # .*?: 匹配任意字符（除了换行符）0次或多次，?表示匹配最少的次数，即非贪婪匹配。
                                       # \]: 匹配右方括号。
    text = re.sub("\\W"," ",text) # 匹配任何非字母、数字、下划线的字符，\是转义字符，因为\W在正则表达式中是特殊字符，需要转义。
    text = re.sub('https?://\S+|www\.\S+', '', text) #re.sub(r'http\S+', '', text)
    text = re.sub('<.*?>+', '', text) #将文本中的HTML标签全部删除，这样可以将一些网页标签信息替换掉。
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) #将文本中的标点符号全部替换成空格，这样可以将一些标点符号替换掉。
    text = re.sub('\n', '', text) #将文本中的换行符全部删除，这样可以将一些换行符替换掉。
    text = re.sub('\w*\d\w*', '', text)     #将文本中的数字全部删除，这样可以将一些数字替换掉。
    return text

def stem_and_filter(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def output_lable(n):
    if n == 1:
        return "Fake News"
    elif n == 0:
        return "Real News"

def manual_testing_voting(input_data):
    testing_news = {"text":[input_data]}
    new_def_test = pd.DataFrame(testing_news)
    
    new_def_test["text"]=new_def_test["text"].apply(preprocess_text)
    new_def_test["text"]=new_def_test["text"].apply(stem_and_filter)

    new_x_test = new_def_test["text"]
    
    
    new_xv_test = loaded_vectorization.transform(new_x_test)
    
    new_xv_test = new_xv_test.toarray()

    # 对新闻进行预测
    pred_voting = loaded_model_voting.predict(new_xv_test)
    pred_LR = loaded_model_LR.predict(new_xv_test)
    pred_DT = loaded_model_DT.predict(new_xv_test)
    pred_RFC = loaded_model_RFC.predict(new_xv_test)
    pred_GBC = loaded_model_GBC.predict(new_xv_test)
    
    # 显示各分类器的预测结果
    st.write("LR Prediction:", output_lable(pred_LR[0]))
    st.write("DT Prediction:", output_lable(pred_DT[0]))
    st.write("RFC Prediction:", output_lable(pred_RFC[0]))
    st.write("GBC Prediction:", output_lable(pred_GBC[0]))
    st.write("Voting Classifier Prediction:", output_lable(pred_voting[0]))

    

  
def app():
    st.title('Fake News Detection')
    st.write('Enter a news article to check if it is fake or real.')
    news = st.text_input('News article', '')
    prediction=''
    if st.button('Check'):
        prediction = manual_testing_voting(news)
       

    

if __name__=='__main__':
    app()

