newsgroups_test = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes'))
X_test = vectorizer.transform(newsgroups_test.data)