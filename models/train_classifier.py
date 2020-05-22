import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine


import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



def load_data(database_filepath):
    """
    A function to load Disaster Response db stored during the ETL process.
    
    INPUT:  
    database_filepath - filepath of disaster respnse db post ETL

    OUTPUT: 
    X - input variable X that has an array of messages
    Y - target variable with classification results for 36 categories 
    category_names - list of 36 classification categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Message", con=engine)
    X = df.message.values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    """
    A function to tokenize and lemmatize message data.
    
    INPUT:  
    text - corpus of messages

    OUTPUT: 
    clean_tokens - returns clean tokens with a message tokenized, lemmatized, case normalized and cleaned for any leading/trailing space
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    A function to build a RandomForestClassifier model using CV. The function first vectorize and applies Tfidf in a pipeline on the messages data
    and then performs multi-output classification on the 36 categories in the dataset. The model then GridSearch to find the best parameters.

    OUTPUT: 
    model - GridSearchCV object that can be used to fit and predict on data.
    """
    
    # specifiy pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search. To reduce execution time, I have tried to keep the best parameters from previous run.
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 2),),
        'features__text_pipeline__vect__max_df': (1.0,),
        'features__text_pipeline__tfidf__use_idf': (True,),
        'clf__estimator__criterion': ['gini'],
        'clf__estimator__n_estimators': [20],
        'clf__estimator__min_samples_split': [2],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
        )       
    }

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=0)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A function to evaluet the machine learning model. The model predicts on test data and then ouputs the classification report for
    36 categories in the dataset.
    
    INPUT:
    model - machine learning model
    X_test - messages from testing data set that we would apply the model to predict the values in 36 categories.
    Y_test - actual values for the 36 categories. This will be used to evaluate the performance of the model.

    """ 
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    # print classification_report for each category. The following measures are printed for each category: f1 score, precision and recall.
    for name in category_names:
        idx = Y_test.columns.get_loc(name)
        print(name)
        print(classification_report(Y_test[name], Y_pred[:,idx]))
    


def save_model(model, model_filepath):
    """
    A function to save the trained model as a pickle file.
    
    INPUT:  
    model - trained model
    model_filepath - location and name of model to be saved.

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()