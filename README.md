# udacity-ds-disaster-response
https://github.com/amygao21/udacity-ds-disaster-response
ml-pipeline project 
# Disaster Response Pipeline Project
### Objective:
In this project, I applied machine learning pipeline skills to analyze disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages. this project used NLP techniques to classify disater messages and sort into 36 categories. The result will be shown on the web app interface.
### Methods:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
https://github.com/amygao21/udacity-ds-disaster-response
