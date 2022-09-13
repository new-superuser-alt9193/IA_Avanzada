from joblib import load

model = load('trainedModel.joblib')

def makePrediction(X):
    y_pred = model.predict(X)
    return y_pred