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

st.header('Wellness Chatbot')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)

    data['similar'] = data['encoding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = data.loc[data['similar'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
