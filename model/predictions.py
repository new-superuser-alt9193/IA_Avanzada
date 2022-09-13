import model2api

model = model2api.generateModel()

def makePrediction(X):
    y_pred = model.predict(X)
    return y_pred