import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
import joblib


def create_model():
    """Trains and saves a model.
    Should be run from `example_projects/nlp_medical_transcript_classifier` dir

    data taken from https://www.kaggle.com/tboyle10/medicaltranscriptions
    """
    # load data
    df = pd.read_csv('datasets/mtsamples.csv').dropna()
    df['medical_specialty'] = df['medical_specialty'].apply(lambda x: x.strip().lower())
    df['transcription'] = df['transcription'].apply(lambda x: x.strip().lower())
    target_labels = ['orthopedic',
        'cardiovascular / pulmonary',
        'radiology',
        'consult - history and phy.',
        'gastroenterology',
        'neurology',
        'general medicine',
        'soap / chart / progress notes',
        'urology',
        'obstetrics / gynecology'
    ]
    df = df[df['medical_specialty'].isin(target_labels)][['transcription', 'medical_specialty']].reset_index(drop=True)
    X = df['transcription']
    y = df['medical_specialty']

    # train vectorizer
    print('vectorizing')
    vec = TfidfVectorizer(min_df=0.001, max_df=0.7, max_features=1000, stop_words='english')
    vec_X = vec.fit_transform(X)

    enc = LabelEncoder()
    enc_y = enc.fit_transform(y)

    # train model
    # Number of trees.
    n_estimators = 1000

    # Define classifier.
    forest_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, max_leaf_nodes=None, class_weight='balanced', oob_score=True, n_jobs=-1, random_state=0)

    # Define grid.
    parameters = {'max_leaf_nodes':np.linspace(20,35,14,dtype='int')}

    # Balanced accuracy as performance measure.
    print('training model')
    clf = RandomizedSearchCV(forest_clf, parameters, n_iter=10, cv=3,iid=False, scoring='accuracy',n_jobs=-1)
    classifier = clf.fit(vec_X, enc_y)

    # Retrieve optimum.
    forest = classifier.best_estimator_

    # Retrieve values
    y_pred = forest.predict(vec_X)
    # Compute scores.
    f1_score_ = f1_score(enc_y, y_pred,average="weighted")
    print('F1 Score', f1_score_)

    pipeline = make_pipeline(vec, forest)

    joblib.dump(pipeline, 'medical_transcript_model/model.pkl')
    joblib.dump(enc, 'medical_transcript_model/label_encoder.pkl')

if __name__ == "__main__":
    create_model()