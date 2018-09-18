lr_cv_classifier = LogisticRegressionCV(multi_class='multinomial',
                                        solver='lbfgs')
lr_cv_classifier.fit(X_train, newsgroups_train.target)
print('Optimal C value', lr_cv_classifier.C_[0])
print('train accuracy', 
     lr_cv_classifier.score(X_train, newsgroups_train.target),
     'test accuracy',
     lr_cv_classifier.score(X_test, newsgroups_test.target))
