from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
voc_sizes = [1500, 3000, 5000]
Cs = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
pipeline = Pipeline([('vectorizer', CountVectorizer(stop_words='english')), 
                     ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs'))])
param_grid = {
        'vectorizer__max_features': voc_sizes,
        'classifier__C': Cs
    }
grid = GridSearchCV(pipeline, 
                   param_grid=param_grid,
                   verbose=2)
grid.fit(newsgroups_train.data, newsgroups_train.target)
print('Best hyperparameters', grid.best_params_)
print('Best train score', grid.best_score_)
print('Test score of best estimator', grid.score(newsgroups_test.data, newsgroups_test.target))