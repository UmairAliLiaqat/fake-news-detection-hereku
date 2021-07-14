from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/check_news', methods=['GET', 'POST'])
def check_news():
    new_model= load_model('fake_news_detection.h5')
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    from keras.preprocessing.sequence import pad_sequences
    maxlen = 1000
    if request.method == 'POST':
        X=request.form.get('news')
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X,maxlen = maxlen)
        pred = (new_model.predict(X) >=0.5).astype(int)
        if pred[0] == 0:
            pred = 'false'
        else:
            pred = 'true'
        return render_template('index.html', pred=pred)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.9',port=4455,debug=True)
