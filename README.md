# Disaster-response-Overview

## Objective
The project will aim at classifying people messages in time of a disaster to classify the importance and the relevance of the messages to disaster response organizations. 
The project will involve 
* creating ETL pipeline to process messages from CSVs. 
* Then load the data to SQLite. 
* ML will read the SQlite database and create multioutput supervised learning model. 
* Finally a webapp will be created to visualize the results.


## Installations, 
The following packages should be downloadd for the text analytics task
* nltk.download('punkt')
* nltk.download('stopwords')
* nltk.download('wordnet')

MultiOutputClassifier should be imported from SKlearn



## file description, 
### app folder 
It has the HTML template file and the **run.py** that has the running instructions of the project.
### data folder 
It has the csv files of the messages and the categories. In addition, it is where the project saves the SQLite databse and it shwere the **process_data.py** is located that makes all the data preparations for the model.
### models
It is where the **train_classifier.py** is saved that has the code that takes the clean dataframe produced by the **process_data.py** and train the multiclassification model then save the resulting model as **classifier.pkl** in the models folder. 



## Running instructions
The project was run as follows:
* Open the terminal
* run: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* run: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
* run: python app/run.py

## Acknowledgments

Thanks to Udacity Data Science Degree team for teaching intesnively and practicly thus i have managed with their help to complete the project.
