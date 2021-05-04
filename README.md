# Disaster-response-
The project will aim at classifying people messages in time of a disaster to classify the importance and the relevance of the messages to disaster response organizations. 
The project will involve 
* creating ETL pipeline to process messages from CSVs. 
* Then load the data to SQLite. 
* ML will read the SQlite database and create multioutput supervised learning model. 
* Finally a webapp will be created to visualize the results.

The project was run as follows:
* Open the terminal
* run: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* run: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
* run: python app/run.py
