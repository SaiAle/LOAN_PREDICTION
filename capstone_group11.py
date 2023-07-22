def ModellingandLinear(X,y):

    from sklearn.model_selection import train_test_split
    X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    from sklearn.model_selection import cross_val_score
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression(solver='lbfgs', max_iter=10000)
    model.fit(X_train,y_train)
    LogisticRegression()
    y_prediction=model.predict(x_test)
    cross_val_result=cross_val_score(model,X,y,cv=5)
    m=np.mean(cross_val_result)
    s=model.score(x_test,y_test)
    return m,s
def gnbmodel(X,y):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    from sklearn.model_selection import train_test_split
    X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(x_test)
    from sklearn import metrics
    s=str(gnb.score(x_test,y_test))
    return print(s)

def knn(X,y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors = 300, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(x_test)
    from sklearn.metrics import confusion_matrix,accuracy_score
    cm = str(confusion_matrix(y_test, y_pred))
    ac = str(accuracy_score(y_test,y_pred))
    s=str(classifier.score(x_test,y_test))  
    return cm,ac,s
