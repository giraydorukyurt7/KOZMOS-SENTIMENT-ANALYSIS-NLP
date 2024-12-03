from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import sys
import os
import pandas as pd
app = Flask(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Kozmos_Sentiment_Analysis_NLP')))
from internal_functions.sentence_to_df import sentence_to_df

# load log model
with open('../../Generated_files/log_final.pkl', 'rb') as f:
    log_model = pickle.load(f)
# load vectorizer
with open('../../Generated_files/tfidf_vectorizer.pkl', 'rb') as f:
    tf_idf_word_vectorizer = pickle.load(f)

@app.route('/')
def main_page():
    return render_template('mainpage.html')

@app.route('/predict_log', methods=['POST'])
def predict_log():
    try:
        user_input = request.form['user_input']
        print(f"User input: {user_input}")

        # TF-IDF dönüşümünü ve tahmini yapıyoruz
        new_sentence_tf_idf, cleaned_sentence = sentence_to_df(user_input, tf_idf_word_vectorizer)
        print(f"Cleaned Sentence: {cleaned_sentence}")
        print(f"TF-IDF Features: {new_sentence_tf_idf}")
        
        # Tahmin yap ve int veri tipine dönüştür
        prediction_log = log_model.predict(new_sentence_tf_idf)
        prediction_log = int(prediction_log[0])  # int32'yi yerleşik int'e dönüştür

        return jsonify({'prediction': prediction_log})  # Tahmin sonucunu JSON olarak döndür
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# http://127.0.0.1:5000/