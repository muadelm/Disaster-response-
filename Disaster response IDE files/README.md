# Data Science Nanodegree
# Udacity

### Disaster Response NLP overview

The project aims at constructing:
* ETL pipeline **[process_data.py]** to extract messages recieved during a disaster then cleaning and wrangling the data in a meaningful formate.
* Load the clean database to SQLite database **[DisasterResponse.db]**.
* ML pipeline **[train_classifier.py]** will read the data from the SQLite to create multi-output classification model **AdaBoostClassifier**. 
* Optimize the model using GridSearsh and save the optimized model to **[classifier.pkl]**.
* A **web app** will produce visualizations and will allow classifying new input messages.


### File discribtion 

- app
* | - template
* | |- master.html  # main page of web app
* | |- go.html  # classification result page of web app
* |- run.py  # Flask file that runs app

- data
* |- disaster_categories.csv  # data to process 
* |- disaster_messages.csv  # data to process
* |- process_data.py
* |- InsertDatabaseName.db   # database to save clean data to

- models
* |- train_classifier.py
* |- classifier.pkl  # saved model 


The project was run as follows after opening the terminal:
* python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
* python app/run.py

After running the files go to **https://view6914b2f4-3001.udacity-student-workspaces.com/** to view the dashboard.
