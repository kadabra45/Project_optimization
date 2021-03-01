from joblib import load
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional
from fastapi import FastAPI, Query

### Utilisation de FastAPI
app = FastAPI()

### Bouton de fastapi permettant d'afficher un message par défaut
@app.post("/Welcome")
async def root():
    return ({
                "Message" : "Bonjour, ceci est la beta d'un algorithm d'analyse de sentiment",
                "Status Code": 200
            })

### Bouton permettant d'effectuer la prédiction de sentiment
@app.put("/Sentiment")
async def root_text(q: str = Query(..., min_length = 1)):

    df = pd.read_csv('comments_train.csv')
    df['comment'].add(q)
    vectorizer_test = TfidfVectorizer()
    X_entrain = vectorizer_test.fit_transform(df['comment'])
    
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X_entrain, y, stratify=y, test_size=0.3, random_state = 1)
    clf_test = LinearSVC()
    clf_test.fit(X_train,y_train)
    predicted_test = clf_test.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_test) 
    precision = precision_score(y_test, predicted_test, average="binary", pos_label="Positive") 
    recall = recall_score(y_test, predicted_test, average="binary", pos_label="Positive") 
    f1_score_met = f1_score(y_test, predicted_test, pos_label="Positive")
    return ({
                    "text" : q,
                    "prediction" : predicted_test[0],
                    "accuracy" : accuracy,
                    "precision" : precision,
                    "recall" : recall,
                    "f1_score" : f1_score_met,
                    "Status Code": 200
            }
            )
