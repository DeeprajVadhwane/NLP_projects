import numpy as np
from flask import Flask, render_template, request
from flask_cors import CORS
import os
import pickle
import newspaper
from newspaper import Article
import urllib
import nltk

# nltk.download('punkt')
nltk.download('punkt')


app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)
   
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        news = article.summary
        pred = model.predict([news])
        return render_template('index.html', prediction_text='The news is "{}"'.format(pred[0]))
    except newspaper.article.ArticleException:
        error_message = 'Failed to download the article. Please provide a valid URL.'
        return render_template('index.html', prediction_text='Error: {}'.format(error_message))
    except Exception as e:
        error_message = str(e)
        return render_template('index.html', prediction_text='An error occurred: {}'.format(error_message))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
