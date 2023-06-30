import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

def model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

def dataset_load():
    data = pd.read_csv('encoding_wellness_data.csv')
    data['encoding'] = data['encoding'].apply(json.loads)
    return data

model = model()
data = dataset_load()

def main():
    st.title('Wellness Chatbot')
    st.header('User profile')

    name = st.text_input('Name')

    if name != '':
        st.subheader(name + '님 안녕하세요!')

    age = st.slider('Age', 1, 100, 30, 5)
    st.text('My age : {}'.format(age))




if __name__ == '__main__':
    main()

