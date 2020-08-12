import pandas as pd
# from edf_complaints import settings
import numpy as np
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import auc, precision_recall_curve
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import shutil
import time
import joblib
import random

# Set directory to save items
models = settings.dir_models

def main():

    # load data
    data = load_breast_cancer(as_frame=True)['frame']
    X, y = data.drop(columns='target'), data['target']

    # Build Pipeline
    scaler = StandardScaler()
    pca = PCA()
    lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000)

    pipe = Pipeline(steps=[
        ('scaler', scaler),
        ('pca', pca),
        ('lr', lr)
    ])

    # Search Space Params
    param_dist = {
        'pca__whiten': [True, False],
        'pca__n_components': [1, 2, 3, 4, 5],
        'lr__C': np.random.uniform(0.001, 1, size=10),
        'lr__l1_ratio': np.random.uniform(0, 1, size=10),
    }

    # Scoring
    scoring = {
        'AUC': 'roc_auc',
        'AUC_PRC': auc_prc_scorer,
        'Precision': 'precision',
        'F1': 'f1',
        'Recall': 'recall'
    }

    # Search
    n_iter_search = 50
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring=scoring,
        refit='F1',
        return_train_score=True,
        cv=5,
        n_jobs=4

    )

    random_search.fit(X, y)
    output = random_search_parse_params(random_search)

    models = settings.dir_base / 'models'

    # Set output folder and make directory if not exists
    outputFolder = models / str(random.getrandbits(64))
    outputFolder.mkdir(exist_ok=True)

    # Save training script, params, model, and data used for analysis
    output.to_json(outputFolder / 'params_scores.json', orient='index')
    joblib.dump(random_search, outputFolder / 'model.pkl')
    data.to_csv(outputFolder / 'data.csv')

    fromFile = Path(__file__)
    toFile = outputFolder / fromFile.name
    shutil.copy(str(fromFile), str(toFile))
    return None


def auc_prc_scorer(model, X, y, **kwargs):
    y_pred_proba = model.predict_proba(X)
    prc_precision, prc_recall, prc_thresholds = precision_recall_curve(y, y_pred_proba[:, 1])
    return auc(prc_recall, prc_precision)


def random_search_parse_params(model):
    res = pd.DataFrame(model.cv_results_)

    res.index.name = 'iteration'
    melt = res.stack().to_frame('value').reset_index(level=1)
    melt.columns = ['metric', 'value']

    melt['metric'] = melt['metric'].str.replace(r'split\d_', '')
    melt = melt.set_index('metric', append=True).groupby(level=[0, 1]).agg(list)
    melt = melt['value'].apply(lambda x: x[0] if len(x) == 1 else x).unstack()
    melt['time'] = time.time()
    melt['model_steps'] = ', '.join([x[0] for x in model.estimator.steps])
    return melt


if __name__ == '__main__':
    main()
