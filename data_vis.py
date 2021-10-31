import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)) 
mse_scorer = make_scorer(rmse)

df = pd.read_csv('data_rand.csv')
y = df['y']
X = df.drop('y',axis=1)
tree = False #true2ならランダムフォレストとか、Falseなら線形回帰系

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state = 0,
    test_size = 0.3
)

if tree:
    #model = ExtraTreesRegressor(n_estimators=100,max_depth=9,min_samples_leaf=4)
    model = RandomForestRegressor()#max_depth=4,min_samples_leaf=3)
    model.fit(X_train,y_train)
    predY0 = model.predict(X_train)
    predY = model.predict(X_test)
    print('train',np.sqrt(mean_squared_error(y_train,predY0)))
    print(r2_score(y_train,predY0))
    print('test',np.sqrt(mean_squared_error(y_test,predY)))
    print(r2_score(y_test,predY))
    plt.figure(figsize=(10,10))
    plt.scatter(y_test,predY)
    plt.scatter(y_train,predY0)
    minY = min(min(y_test),min(predY))*0.9
    maxY = max(max(y_test),max(predY))*1.1
    plt.xlim([minY,maxY])
    plt.ylim([minY,maxY])
    plt.plot([minY,maxY],[minY,maxY])
    plt.show()

    #permutation importance
    result = permutation_importance(model, X_train, y_train, scoring=mse_scorer, n_repeats=10, n_jobs=-1, random_state=71)
    plt.scatter(list(range(100)),result["importances_mean"],label="importances_mean")
    plt.scatter(list(range(100)),result["importances_std"],label="importances_std")
    plt.legend()
    plt.show()

    n = 2 #トライ回数
    importance = [0]*100
    for i in range(n):
        #model = ExtraTreesRegressor(n_estimators=100,max_depth=9,min_samples_leaf=4)
        model = RandomForestRegressor()#max_depth=4,min_samples_leaf=3)
        model.fit(X,y)
        imp = model.feature_importances_
        for j in range(100):
            importance[j] += imp[j]
    plt.scatter(list(range(100)),importance)
    plt.show()

else:
    model = LinearRegression()
    model.fit(X_train,y_train)
    predY0 = model.predict(X_train)
    predY = model.predict(X_test)
    print('train',np.sqrt(mean_squared_error(y_train,predY0)))
    print(r2_score(y_train,predY0))
    print('test',np.sqrt(mean_squared_error(y_test,predY)))
    print(r2_score(y_test,predY))
    plt.figure(figsize=(10,10))
    plt.scatter(y_test,predY)
    plt.scatter(y_train,predY0)
    minY = min(min(y_test),min(predY))*0.9
    maxY = max(max(y_test),max(predY))*1.1
    plt.xlim([minY,maxY])
    plt.ylim([minY,maxY])
    plt.plot([minY,maxY],[minY,maxY])
    plt.show() 

    #permutation importance
    result = permutation_importance(model, X_train, y_train, scoring=mse_scorer, n_repeats=10, n_jobs=-1, random_state=71)
    plt.scatter(list(range(100)),result["importances_mean"],label="importances_mean")
    plt.scatter(list(range(100)),result["importances_std"],label="importances_std")
    plt.legend()
    plt.show()

    model = LinearRegression()
    model.fit(X,y)
    val = model.coef_
    plt.scatter(list(range(100)),val)
    plt.show()
