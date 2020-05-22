import json
import plotly
import pandas as pd
import numpy as np
import re
from collections import Counter

import nltk
nltk.download(['stopwords', 'punkt'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for second graph
    dist = []
    names = []
    for idx, name in enumerate(df.columns):
        if idx>3:
            names.append(name)
            dist.append(df[name].sum())
    
    # data for third graph
    # calculate word frequency
    words = df.message.str.lower()
    words = pd.DataFrame(words.str.split(' '))
    # create list of words
    words_list = np.unique(np.hstack(words.message)).tolist()
    # clean up data
    words_list = [re.sub('[^0-9A-Za-z!?]+', ' ', x) for x in words_list]
    words_list = [re.sub('!','',x) for x in words_list]                                   
    words_list = [x for x in words_list if len(x) > 2]
    words_list = [x.strip(' ') for x in words_list]
    
    # set of stopwords from nltk
    stopwords = nltk.corpus.stopwords.words('english')
    # remove stopwords
    words_list = [x for x in words_list if x not in stopwords]
    
    words_counter = Counter(words_list)
    words_df = pd.DataFrame.from_dict(words_counter, orient='index')
    words_df.rename(columns={0: 'count'}, inplace=True)
    words_df.sort_values('count', ascending=False, inplace=True)
    words_df = words_df.head(20)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },   
        {
            'data': [
                Bar(
                    x=names,
                    y=dist
                )
            ],

            'layout': {
                'title': 'Distribution of 36 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words_df['count'].index,
                    y=words_df['count'].values
                )
            ],

            'layout': {
                'title': 'Top 20 words in Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()