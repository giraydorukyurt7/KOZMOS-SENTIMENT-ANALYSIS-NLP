from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# load model
with open('log_final.pkl', 'rb') as f:
    log_model = pickle.load(f)

@app.route('/')
def main_page():
    return render_template('mainpage.html')

@app.route('/predict_log', methods=['POST'])
def predict_log():
    user_input = request.form['user_input']
    prediction_log = log_model.predict([user_input])
    return jsonify({'prediction': prediction_log[0]})

if __name__ == '__main__':
    app.run(debug=True)    