from vectorizer import TFIDFVectorizer,CountVectorizer
from naiveBayes import CategoricalNB,GaussianNB,NaiveBayes
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import streamlit as st
import string 
import warnings
warnings.filterwarnings(action='ignore')

sms_spam = \
pd.read_csv('SMSSpamCollection', sep='\t', header=None, \
            names=['Label', 'SMS'])

#print(sms_spam.shape)

punctuation = string.punctuation
sms_spam['SMS'] = sms_spam['SMS'].apply(lambda x: str.translate(str.lower(x),str.maketrans('','',punctuation)))
sms_spam['Label'] = sms_spam['Label'].replace({'ham':0,'spam':1})

st.title('SMS Spam Detection App')

sms = CountVectorizer()
X = sms.fit_transform(sms_spam['SMS'])
y = sms_spam['Label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)

nb = NaiveBayes()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)

accurecy = sum(y_test ==y_pred)/len(y_test)
#print(accurecy) 

user_input = st.text_area('Enter your SMS here:')
if st.button('Predict'):
    new_sms = sms.transform([user_input])
    prediction = nb.predict(new_sms)[0]
    if prediction == 0:
        st.success('This SMS is not spam.')
    else:
        st.error('Warning! This SMS is spam.')