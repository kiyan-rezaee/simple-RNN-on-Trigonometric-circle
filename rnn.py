import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

def sin():
    t = np.arange(0, 2000)
    X = np.sin(0.01 * t)
    return X

def cosine():
    t = np.arange(0, 2000)
    X = np.cos(0.01 * t)
    return X

def tan():
    t = np.arange(0, 2000)
    X = np.tan(0.01 * t)
    return X

def tanh():
    t = np.arange(0, 2000)
    X = np.tanh(0.01 * t)
    return X  

def makedataset(X):
    X_train, X_test = X[:1500], X[1500:]
    XT, yT, Xt, yt = [], [], [], []
    for i in range(len(X_train) - 15):
        d = i + 15
        XT.append(X_train[i:d,])
        yT.append(X_train[d])
    for i in range(len(X_test) - 15):
        d = i + 15
        Xt.append(X_test[i:d,])
        yt.append(X_test[d])
    
    XT = np.array(XT)
    Xt = np.array(Xt)
    yT = np.array(yT)
    yt = np.array(yt)
    # make it ready for model
    XT = np.reshape(XT, (XT.shape[0], XT.shape[1], 1))
    Xt = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 1))
    return [XT, yT, Xt, yt]

def model(X, XT, yT, Xt, yt):
    model = Sequential()
    model.add(SimpleRNN(units=64, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    model.fit(XT, yT, epochs=100, batch_size=20)
    print(model.evaluate(Xt, yt))
    XTPredicted = model.predict(XT)
    XtPredicted = model.predict(Xt)
    XFinal = np.concatenate([XTPredicted, XtPredicted], axis=0)
    plt.plot(X, color='red')
    plt.plot(XFinal, color='blue')
    plt.show()

while True:
    print('1.sine    2.cosine    3.tangent   4.tanh')
    input = input()
    if int(input) == 1:
        model(sin(), makedataset(sin())[0], makedataset(sin())[1], makedataset(sin())[2], makedataset(sin())[3])
        break
    if int(input) == 2:
        model(cosine(), makedataset(cosine())[0], makedataset(cosine())[1], makedataset(cosine())[2], makedataset(cosine())[3])
        break
    if int(input) == 3:
        model(tan(), makedataset(tan())[0], makedataset(tan())[1], makedataset(tan())[2], makedataset(tan())[3])
        break
    if int(input) == 4:
        model(tanh(), makedataset(tanh())[0], makedataset(tanh())[1], makedataset(tanh())[2], makedataset(tanh())[3])
        break        
