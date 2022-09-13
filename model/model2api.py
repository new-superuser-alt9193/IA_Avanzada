from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd

def generateModel():
    df = pd.read_csv("../data/train_user_collapsed_small_data.csv")
    x = df.drop(columns=['customer_ID', 'target'])
    y = df['target']

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state= 1)

    clf = MLPClassifier(hidden_layer_sizes=(80, 65), max_iter=250, activation = 'relu', solver = 'adam', random_state=1, learning_rate_init= 0.001)

    clf.fit(x_test, y_test)

    return clf

clf = generateModel()
dump(clf, './trainedModel.joblib')

