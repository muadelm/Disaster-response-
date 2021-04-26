import sys

import pandas as pd
import numpy as np
import re
import nltk
import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import multioutput
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier


def load_data(database_filepath):
    """
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    df = pd.merge(messages, categories, on="id")
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply (lambda x: x.rstrip ('- 1 0'))
    categories.columns = category_colnames
    
    # Iterate over columns to extract the number only from all entries
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    df = df.drop(axis=1, columns='categories')
    #add the extracted categories columns into the dataframe and drop the missed category column
    df = pd.concat([df, categories], axis =1)
    
    ## drop dublicated entries from the dataframe
    df.drop_duplicates(inplace = True)
    
    # extract the data to 
    engine = create_engine('sqlite:///disaster_res.db')
    df.to_sql('disaster_res', engine, index=False)
    """
    engine = create_engine(database_filepath)
    df = pd.read_sql ('SELECT * FROM DisasterResponse', engine)
    X = df.message
    Y = df.iloc[:,4:]


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    stop_words = stopwords.words("english")
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    parameters_grid = {'clf__estimator__learning_rate': [0.5, 1],
              'clf__estimator__n_estimators': [30, 60]}
    cv = GridSearchCV(pipeline, param_grid=parameters_grid)
    cv.fit(X_train, y_train)


def evaluate_model(model, X_test, Y_test, category_names):
    ytest_pred_cv = cv.predict(X_test)
    ytest_pred_cv = pd.DataFrame (y_pred, columns = category_names)
    for column in category_names:
        print('Metrics of teh classification pipeline on ""', column, '"" outcome is :' )
        print(metrics_class(column, Y_test, ytest_pred_cv))


def save_model(model, model_filepath):
    ext = pickle.dumps(model_filepath)


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