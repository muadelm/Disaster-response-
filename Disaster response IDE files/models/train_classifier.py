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
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Input:
        database path that has the cleaned data from the ETL
    Output:
        X that has the meassages
        Y that has the features
        category_names that has the names of features
    """
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql ('SELECT * FROM DisasterResponse', engine)
    X = df.message
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    """
    Input:
        text: original text for messages
    Output:
        clean_tokens: tokenized text for model
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    words = [w for w in words if w not in stopwords.words('english')]

    clean_tokens = []
    for w in words:
        clean_tok = lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    #X_train, X_test, y_train, y_test = train_test_split(X, Y)
    parameters_grid = {'clf__estimator__n_estimators': [60]}
    model = GridSearchCV(pipeline, param_grid=parameters_grid)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Y_pred = model.predict(X_test)
    def metrics_class(col, y_actual, y_calc):
        return classification_report(y_actual[col], y_calc[col])

    Y_pred = pd.DataFrame (Y_pred, columns = category_names)
    for col in category_names:
        print('Metrics of teh classification pipeline on ""', column, '"" outcome is :' )
        print(metrics_class(column, y_test, y_pred))target_names=category_names))
    """
    Y_pred = model.predict(X_test)
    # print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), target_names=category_names))
    report = classification_report(Y_test,Y_pred,target_names = category_names)
    return report

def save_model(model, model_filepath):
    #pickle.dumps(model_filepath)
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
