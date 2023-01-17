import pickle
import streamlit as st
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
import logging
import tensorflow as tf

class enfrtrans():
    def __init__(self, model_path):
        self.model = load_model(model_path)
        with open("tokenizer_eng.pkl", 'rb') as handle:
            self.tokenizer_eng = pickle.load(handle)
        with open('tokenizer_fr.pkl', 'rb') as handle:
            self.tokenizer_fr = pickle.load(handle)
            
            
    def translate(self, eng_sentence):
        eng_max_length = 15
        fr_max_length = 21
        sentence_token = self.tokenizer_eng.texts_to_sequences([eng_sentence])
        sentence_pad = pad_sequences(sentence_token, maxlen=eng_max_length, padding='post')
        prediction = self.model.predict(sentence_pad)
        translated_words = tf.argmax(prediction, axis=2)
        no_zero = [i for i in translated_words[0] if i != 0]
        translated_sentence = [list(self.tokenizer_fr.word_index)[word - 1] for word in no_zero]
        translated_sentence = ' '.join(translated_sentence)
        return translated_sentence
    
def main():
    st.title("English To French Translator")
    model = enfrtrans('2BiLSTM.h5')
    eng_sentence = st.text_input("Enter an english text: ")
    if eng_sentence:
        predicted_class = model.translate(eng_sentence)
        st.success(predicted_class)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()