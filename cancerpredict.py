import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():

    #read data
    df_train = pd.read_csv('/Users/kerwinzhou/Downloads/Python_ML_and_Kaggle-master/Datasets/Breast-Cancer/breast-cancer-train.csv')
    df_test = pd.read_csv('/Users/kerwinzhou/Downloads/Python_ML_and_Kaggle-master/Datasets/Breast-Cancer/breast-cancer-test.csv')
    print(df_train.head(10))
    print(df_test.head(10))

    df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
    df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

    plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker = 'o', s=200,c='red')
    plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')

    b = np.random.random([1])
    coef = np.random.random([2])
    lx = np.arange(0, 12)
    ly = (- b - lx * coef[0])/coef[1]
    plt.plot(lx, ly, c='yellow')
    plt.show()

    lr = LogisticRegression()

    lr.fit(df_train[['Clump Thickness','Cell Size']], df_train['Type'])

    intercept = lr.intercept_
    coef = lr.coef_[0,:]

    ly = (-intercept-lx*coef[0])/coef[1]

    plt.plot(lx, ly, 'green')
    plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='o', s=200, c='red')
    plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='x', s=150, c='black')
    plt.xlabel('Clump Thickness')
    plt.ylabel('Cell Size')
    plt.show()



if __name__ == '__main__':
    main()