from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle


xgb = pickle.load(open('xgboost_model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('quora_index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['review_text']
        text = [message]
        print(text)
        test_features = vectorizer.transform(text)
        df_text = pd.DataFrame(test_features.toarray(), columns=vectorizer.get_feature_names())
        my_prediction = xgb.predict(df_text)
        print(my_prediction)
        if my_prediction==0:
            prediction = 'NEGATIVE'
        elif my_prediction == 1:
            prediction = 'POSITIVE'
        print(prediction)
    return render_template('predict.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)