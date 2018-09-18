lr_classifier.fit(X_train, newsgroups_train.target)
print('train accuracy', 
     lr_classifier.score(X_train, newsgroups_train.target),
     'test accuracy',
     lr_classifier.score(X_test, newsgroups_test.target))
