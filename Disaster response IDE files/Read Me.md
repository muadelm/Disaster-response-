#Data Science Nanodegree
#Udacity

#Disaster Response NLP

The project aims at constructing:
* ETL pipeline **[process_data.py]** to extract messages recieved during a disaster then cleaning and wrangling the data in a meaningful formate.
* Load the clean database to SQLite database **[DisasterResponse.db]**.
* ML pipeline **[train_classifier.py]** will read the data from the SQLite to create multi-output classification model **AdaBoostClassifier**. 
* Optimize the model using GridSearsh and save the optimized model to **[classifier.pkl]**.
* A **web app** will produce visualizations and will allow classifying new input messages.
