import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def main():
    titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    #print(titanic.head())
    #titanic.info()
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    #X.info()

    X['age'].fillna(X['age'].mean(), inplace = True)
    #X.info()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
    X_test = vec.transform(X_test.to_dict(orient ='record'))

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_predict = dtc.predict(X_test)

    print(dtc.score(X_test, y_test))
    print(classification_report(y_predict, y_test, target_names=['died', 'survived']))


if __name__ == '__main__':
    main()

