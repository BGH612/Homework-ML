import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import SGDRegressor
df=pd.read_csv('E:\mag\python\car_price_w.csv')
#print(df)
# для своего дз я буду использовать данные, которые уже были обработаны мной в прошлом домашнем задании, по данным уже был произведен data cleaning, выполнен разведочный анализ, feature engineering.

# для начала необходимо разбить выборку на тестовую и тренировочную

y=df['price']
X=df[['transmission','brand_cat','milage','age','fuel_cat','ext_col_cat','int_col_cat','hp','accident']]
b=df[['brand_cat','milage','age','hp','accident']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=324)
#print(y_train)
#масштабируем признаки

scaler=MinMaxScaler(feature_range=(0,1))
#print(X_train)

X_test_scale=scaler.fit_transform(X_test)

X_train_scale=scaler.fit_transform(X_train)

X_train_scaled=pd.DataFrame(X_train_scale,columns=X.columns)
X_test_scaled=pd.DataFrame(X_test_scale,columns=X.columns)

# теперь подберем нилучше количество признаков, используя recursive feature elimination


from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=KFold(5), scoring='accuracy')
rfecv.fit(X_train_scaled, y_train)

selected_features = list(X_train_scaled.columns[rfecv.support_])

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % selected_features)

# оптимальнок количество 9, необходимо использовать все переменные

#Градиентный спуск
def gradient_descent(X1, y1, learning_rate=0.01, num_iterations=1000, lambda_reg=0.1):
    
    X=np.array(X1)
    y=np.array(y1)
    m, n = X.shape  # m - количество примеров, n - количество признаков
    w = np.zeros(n)  # Инициализация весов в нулях

    for i in range(num_iterations):
        # Предсказание
        y_pred = X.dot(w)
        
        # Вычисление ошибки
        error = y_pred - y
        
        # Вычисление градиента (с учетом L2-регуляризации)
        gradient = (2/m) * X.T.dot(error) + (2 * lambda_reg * w)

        # Обновление весов
        w -= learning_rate * gradient

    return w



grad1 = gradient_descent(X_train_scaled, y_train, learning_rate=0.01, num_iterations=1000, lambda_reg=0.1)  # выбранная модель

y_train_pred = X_train_scaled.dot(grad1)

y_test_pred = X_test_scaled.dot(grad1)

# Оценка качества модели
mse_train_grad = mean_squared_error(y_train, y_train_pred)
rmse_train_grad = np.sqrt(mse_train_grad)
r2_train_grad = r2_score(y_train, y_train_pred)

mse_test_grad = mean_squared_error(y_test, y_test_pred)
rmse_test_grad = np.sqrt(mse_test_grad)
r2_test_grad = r2_score(y_test, y_test_pred)

print("Train MSE grad:", mse_train_grad)
print("Train RMSE grad:", rmse_train_grad)
print("Train R² grad:", r2_train_grad)

print("Test MSE grad:", mse_test_grad)
print("Test RMSE grad:", rmse_test_grad)
print("Test R² grad:", r2_test_grad)




# подберем гиперпараметры для ridge
alpha_grid = np.logspace(-3, 3, 10)
searcher = GridSearchCV(Ridge(), [{"alpha": alpha_grid}], scoring="neg_root_mean_squared_error", cv=4)
searcher.fit(X_train_scaled, y_train)


best_alpha = searcher.best_params_["alpha"]
print("Best alpha = %.4f" % best_alpha)


#модель ridge
model = Ridge(alpha=best_alpha)  # выбранная модель

model.fit(X_train_scaled, y_train)  # обучение модели на обучающей выборке

y_train_pred = model.predict(X_train_scaled)  # использование модели для предсказания на обучающей
y_test_pred = model.predict(X_test_scaled)  # или на тестовой выборке

mse_test_ridge= mean_squared_error(y_test, y_test_pred, squared=False)
mse_train_ridge= mean_squared_error(y_train, y_train_pred, squared=False)
rmse_train_ridge = np.sqrt(mse_train_ridge)
rmse_test_ridge = np.sqrt(mse_test_ridge)
r2_train_ridge = r2_score(y_train, y_train_pred)
r2_test_ridge = r2_score(y_test, y_test_pred)
print("mse_test ridge- %.4f" %  mse_test_ridge )
print("mse_train ridge- %.4f" %  mse_train_ridge )
print("rmse_test ridge- %.4f" %  rmse_test_ridge )
print("rmse_train ridge- %.4f" %  rmse_train_ridge)
print("r2_test ridge- %.4f" %  r2_test_ridge )
print("r2_train ridge- %.4f" %  r2_train_ridge)

#используем кросс валидацию, найдем результаты для количества фолдов от 2 до 10


cv_rmse_train2 = cross_val_score(model, X_train_scaled, y_train, cv=2, scoring="neg_root_mean_squared_error")
cv_rmse_test2 = cross_val_score(model, X_test_scaled, y_test, cv=2, scoring="neg_root_mean_squared_error")
cv_mse_train2 = cross_val_score(model, X_train_scaled, y_train, cv=2, scoring="neg_mean_squared_error")
cv_mse_test2= cross_val_score(model, X_test_scaled, y_test, cv=2, scoring="neg_mean_squared_error")
cv_r2_train2 = cross_val_score(model, X_train_scaled, y_train, cv=2, scoring="r2")
cv_r2_test2 = cross_val_score(model, X_test_scaled, y_test, cv=2, scoring="r2")

mean_rmse_train2=np.mean(-cv_rmse_train2)
mean_rmse_test2=np.mean(-cv_rmse_test2)
mean_mse_train2=np.mean(-cv_mse_train2)
mean_mse_test2=np.mean(-cv_mse_test2)
mean_r2_train2=np.mean(cv_r2_train2)
mean_r2_test2=np.mean(cv_r2_test2)

cv_rmse_train3 = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="neg_root_mean_squared_error")
cv_rmse_test3 = cross_val_score(model, X_test_scaled, y_test, cv=3, scoring="neg_root_mean_squared_error")
cv_mse_train3 = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="neg_mean_squared_error")
cv_mse_test3= cross_val_score(model, X_test_scaled, y_test, cv=3, scoring="neg_mean_squared_error")
cv_r2_train3 = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="r2")
cv_r2_test3 = cross_val_score(model, X_test_scaled, y_test, cv=3, scoring="r2")

mean_rmse_train3=np.mean(-cv_rmse_train3)
mean_rmse_test3=np.mean(-cv_rmse_test3)
mean_mse_train3=np.mean(-cv_mse_train3)
mean_mse_test3=np.mean(-cv_mse_test3)
mean_r2_train3=np.mean(cv_r2_train3)
mean_r2_test3=np.mean(cv_r2_test3)

cv_rmse_train4 = cross_val_score(model, X_train_scaled, y_train, cv=4, scoring="neg_root_mean_squared_error")
cv_rmse_test4 = cross_val_score(model, X_test_scaled, y_test, cv=4, scoring="neg_root_mean_squared_error")
cv_mse_train4 = cross_val_score(model, X_train_scaled, y_train, cv=4, scoring="neg_mean_squared_error")
cv_mse_test4= cross_val_score(model, X_test_scaled, y_test, cv=4, scoring="neg_mean_squared_error")
cv_r2_train4 = cross_val_score(model, X_train_scaled, y_train, cv=4, scoring="r2")
cv_r2_test4 = cross_val_score(model, X_test_scaled, y_test, cv=4, scoring="r2")

mean_rmse_train4=np.mean(-cv_rmse_train4)
mean_rmse_test4=np.mean(-cv_rmse_test4)
mean_mse_train4=np.mean(-cv_mse_train4)
mean_mse_test4=np.mean(-cv_mse_test4)
mean_r2_train4=np.mean(cv_r2_train4)
mean_r2_test4=np.mean(cv_r2_test4)
    

cv_rmse_train5 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_root_mean_squared_error")
cv_rmse_test5 = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring="neg_root_mean_squared_error")
cv_mse_train5 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error")
cv_mse_test5 = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring="neg_mean_squared_error")
cv_r2_train5 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
cv_r2_test5 = cross_val_score(model, X_test_scaled, y_test, cv=5, scoring="r2")



print("Cross validation scores:\n\t", "\n\t".join("%.4f" % -x for x in cv_rmse_train5))
mean_rmse_train5=np.mean(-cv_rmse_train5)
mean_rmse_test5=np.mean(-cv_rmse_test5)
mean_mse_train5=np.mean(-cv_mse_train5)
mean_mse_test5=np.mean(-cv_mse_test5)
mean_r2_train5=np.mean(cv_r2_train5)
mean_r2_test5=np.mean(cv_r2_test5)
print("Mean CV RMSE = %.4f" % np.mean(-cv_rmse_train5))
print("STD CV RMSE = %.4f" % np.std(-cv_rmse_train5))
#6 фолдов
cv_rmse_train6 = cross_val_score(model, X_train_scaled, y_train, cv=6, scoring="neg_root_mean_squared_error")
cv_rmse_test6 = cross_val_score(model, X_test_scaled, y_test, cv=6, scoring="neg_root_mean_squared_error")
cv_mse_train6 = cross_val_score(model, X_train_scaled, y_train, cv=6, scoring="neg_mean_squared_error")
cv_mse_test6= cross_val_score(model, X_test_scaled, y_test, cv=6, scoring="neg_mean_squared_error")
cv_r2_train6 = cross_val_score(model, X_train_scaled, y_train, cv=6, scoring="r2")
cv_r2_test6 = cross_val_score(model, X_test_scaled, y_test, cv=6, scoring="r2")

mean_rmse_train6=np.mean(-cv_rmse_train6)
mean_rmse_test6=np.mean(-cv_rmse_test6)
mean_mse_train6=np.mean(-cv_mse_train6)
mean_mse_test6=np.mean(-cv_mse_test6)
mean_r2_train6=np.mean(cv_r2_train6)
mean_r2_test6=np.mean(cv_r2_test6)
# 7 фолдов
cv_rmse_train7 = cross_val_score(model, X_train_scaled, y_train, cv=7, scoring="neg_root_mean_squared_error")
cv_rmse_test7 = cross_val_score(model, X_test_scaled, y_test, cv=7, scoring="neg_root_mean_squared_error")
cv_mse_train7 = cross_val_score(model, X_train_scaled, y_train, cv=7, scoring="neg_mean_squared_error")
cv_mse_test7= cross_val_score(model, X_test_scaled, y_test, cv=7, scoring="neg_mean_squared_error")
cv_r2_train7 = cross_val_score(model, X_train_scaled, y_train, cv=7, scoring="r2")
cv_r2_test7 = cross_val_score(model, X_test_scaled, y_test, cv=7, scoring="r2")

mean_rmse_train7=np.mean(-cv_rmse_train7)
mean_rmse_test7=np.mean(-cv_rmse_test7)
mean_mse_train7=np.mean(-cv_mse_train7)
mean_mse_test7=np.mean(-cv_mse_test7)
mean_r2_train7=np.mean(cv_r2_train7)
mean_r2_test7=np.mean(cv_r2_test7)

#8 фолдов
cv_rmse_train8 = cross_val_score(model, X_train_scaled, y_train, cv=8, scoring="neg_root_mean_squared_error")
cv_rmse_test8 = cross_val_score(model, X_test_scaled, y_test, cv=8, scoring="neg_root_mean_squared_error")
cv_mse_train8 = cross_val_score(model, X_train_scaled, y_train, cv=8, scoring="neg_mean_squared_error")
cv_mse_test8= cross_val_score(model, X_test_scaled, y_test, cv=8, scoring="neg_mean_squared_error")
cv_r2_train8 = cross_val_score(model, X_train_scaled, y_train, cv=8, scoring="r2")
cv_r2_test8 = cross_val_score(model, X_test_scaled, y_test, cv=8, scoring="r2")

mean_rmse_train8=np.mean(-cv_rmse_train8)
mean_rmse_test8=np.mean(-cv_rmse_test8)
mean_mse_train8=np.mean(-cv_mse_train8)
mean_mse_test8=np.mean(-cv_mse_test8)
mean_r2_train8=np.mean(cv_r2_train8)
mean_r2_test8=np.mean(cv_r2_test8)

#9 фолдов
cv_rmse_train9 = cross_val_score(model, X_train_scaled, y_train, cv=9, scoring="neg_root_mean_squared_error")
cv_rmse_test9 = cross_val_score(model, X_test_scaled, y_test, cv=9, scoring="neg_root_mean_squared_error")
cv_mse_train9 = cross_val_score(model, X_train_scaled, y_train, cv=9, scoring="neg_mean_squared_error")
cv_mse_test9= cross_val_score(model, X_test_scaled, y_test, cv=9, scoring="neg_mean_squared_error")
cv_r2_train9 = cross_val_score(model, X_train_scaled, y_train, cv=9, scoring="r2")
cv_r2_test9 = cross_val_score(model, X_test_scaled, y_test, cv=9, scoring="r2")

mean_rmse_train9=np.mean(-cv_rmse_train9)
mean_rmse_test9=np.mean(-cv_rmse_test9)
mean_mse_train9=np.mean(-cv_mse_train9)
mean_mse_test9=np.mean(-cv_mse_test9)
mean_r2_train9=np.mean(cv_r2_train9)
mean_r2_test9=np.mean(cv_r2_test9)

#10 фолдов
cv_rmse_train10 = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring="neg_root_mean_squared_error")
cv_rmse_test10 = cross_val_score(model, X_test_scaled, y_test, cv=10, scoring="neg_root_mean_squared_error")
cv_mse_train10 = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring="neg_mean_squared_error")
cv_mse_test10= cross_val_score(model, X_test_scaled, y_test, cv=10, scoring="neg_mean_squared_error")
cv_r2_train10 = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring="r2")
cv_r2_test10 = cross_val_score(model, X_test_scaled, y_test, cv=10, scoring="r2")

mean_rmse_train10=np.mean(-cv_rmse_train10)
mean_rmse_test10=np.mean(-cv_rmse_test10)
mean_mse_train10=np.mean(-cv_mse_train10)
mean_mse_test10=np.mean(-cv_mse_test10)
mean_r2_train10=np.mean(cv_r2_train10)
mean_r2_test10=np.mean(cv_r2_test10)





data = {
    'Fold2': [mean_rmse_train2, mean_rmse_test2, mean_mse_train2, mean_mse_test2, mean_r2_train2, mean_r2_test2],
    'Fold3': [mean_rmse_train3, mean_rmse_test3, mean_mse_train3, mean_mse_test3, mean_r2_train3, mean_r2_test3],
    'Fold4': [mean_rmse_train4, mean_rmse_test4, mean_mse_train4, mean_mse_test4, mean_r2_train4, mean_r2_test4],
    'Fold5': [mean_rmse_train5, mean_rmse_test5, mean_mse_train5, mean_mse_test5, mean_r2_train5, mean_r2_test5],
    'Fold6': [mean_rmse_train6, mean_rmse_test6, mean_mse_train6, mean_mse_test6, mean_r2_train6, mean_r2_test6],
    'Fold7': [mean_rmse_train7, mean_rmse_test7, mean_mse_train7, mean_mse_test7, mean_r2_train7, mean_r2_test7],
    'Fold8': [mean_rmse_train8, mean_rmse_test8, mean_mse_train8, mean_mse_test8, mean_r2_train8, mean_r2_test8],
    'Fold9': [mean_rmse_train9, mean_rmse_test9, mean_mse_train9, mean_mse_test9, mean_r2_train9, mean_r2_test9],
    'Fold10': [mean_rmse_train10, mean_rmse_test10, mean_mse_train10, mean_mse_test10, mean_r2_train10, mean_r2_test10]
}
columns=['Fold5','Fold4']
result=['cv_rmse_train','cv_rmse_test']
cv_df=pd.DataFrame(data,index=['rmse-train', 'rmse-test', 'mse-train', 'mse-test', 'r2-train', 'r2-test'])
mean_values = cv_df.mean(axis=1)
std_values = cv_df.std(axis=1)

# Добавляем E и STD в таблицу
cv_df['E'] = mean_values
cv_df['STD'] = std_values
cv_df = cv_df.astype(float)

cv_df = cv_df.applymap(lambda x: f"{x:.6f}")

print(cv_df)

# из таблицы, можно увидеть, что для rmse ошибки существенно снижаются начиная с 5 фолдов, это проявляется как для тестовой, так и для тренировочной выборки. 
# Что касается MSE, тут наилучшим образом себя показала модель с двумя фолдами для тестовой и обучающей выборки
# Для R2 ситуауция иная, результаты здесь неоднозначны, сначала начиная с четырех фолдов модель показывает все лучшие результаты, однако начиная с 7-ми фолдов ситуация ухудшается на тестовой выборке
# таким образом наиболее оптимальным вариантом будет выбирать модель с 5ью фолдами кросс валидации 

mse_test= mean_squared_error(y_test, y_test_pred, squared=False)
mse_train= mean_squared_error(y_train, y_train_pred, squared=False)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []





# стохастический градиентный спуск
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, x1, y1):
        X=np.array(x1)
        y=np.array(y1)
        # Инициализация весов
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Проход по всем эпохам
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Предсказание
                y_predicted = np.dot(X[i], self.w) + self.b

                # Вычисление градиентов
                dw = (1 / n_samples) * (y_predicted - y[i]) * X[i]
                db = (1 / n_samples) * (y_predicted - y[i])

                # Обновление весов
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


modelSDG=LinearRegressionSGD(learning_rate=0.01, n_epochs=1000)
modelSDG.fit(X_train_scaled, y_train)

y_train_pred = modelSDG.predict(X_train_scaled)  # использование модели для предсказания на обучающей
y_test_pred = modelSDG.predict(X_test_scaled)  # на тестовой

mse_test_SDG= mean_squared_error(y_test, y_test_pred, squared=False)
mse_train_SDG= mean_squared_error(y_train, y_train_pred, squared=False)
rmse_train_SDG = np.sqrt(mse_train_SDG)
rmse_test_SDG = np.sqrt(mse_test_SDG)
r2_train_SDG = r2_score(y_train, y_train_pred)
r2_test_SDG = r2_score(y_test, y_test_pred)
print("mse_test SDG- %.4f" %  mse_test_SDG )
print("mse_train SDG- %.4f" %  mse_train_SDG )
print("rmse_test SDG- %.4f" %  rmse_test_SDG )
print("rmse_train SDG- %.4f" %  rmse_train_SDG )
print("r2_test SDG- %.4f" %  r2_test_SDG )
print("r2_train SDG- %.4f" %  r2_train_SDG)

# мини градиентный спуск
class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32, n_iterations=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.theta = None

    def fit(self, x1, y1):
        X=np.array(x1)
        y=np.array(y1)
        m, n = X.shape
        self.theta = np.zeros(n)
        
        for iteration in range(self.n_iterations):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                gradients = -2 / self.batch_size * X_batch.T.dot(y_batch - X_batch.dot(self.theta))
                self.theta -= self.learning_rate * gradients

    def predict(self, X):
        return X.dot(self.theta)



# Обучение модели
modelBatch = MiniBatchGradientDescent(learning_rate=0.01, batch_size=32, n_iterations=1000)
modelBatch.fit(X_train_scaled, y_train)

y_train_pred = modelBatch.predict(X_train_scaled) 
y_test_pred = modelBatch.predict(X_test_scaled)  

mse_test_mini= mean_squared_error(y_test, y_test_pred, squared=False)
mse_train_mini= mean_squared_error(y_train, y_train_pred, squared=False)
rmse_train_mini = np.sqrt(mse_train_mini)
rmse_test_mini = np.sqrt(mse_test_mini)
r2_train_mini = r2_score(y_train, y_train_pred)
r2_test_mini = r2_score(y_test, y_test_pred)
print("mse_test miniSDG- %.4f" %  mse_test_mini )
print("mse_train miniSDG- %.4f" %  mse_train_mini )
print("rmse_test miniSDG- %.4f" %  rmse_test_mini )
print("rmse_train miniSDG- %.4f" %  rmse_train_mini )
print("r2_test miniSDG- %.4f" %  r2_test_mini )
print("r2_train miniSDG- %.4f" %  r2_train_mini)


# построим таблицу для сравнения всех моделей
data = {
    'Gradient_descent': [rmse_train_grad, rmse_test_grad, mse_train_grad, mse_test_grad, r2_train_grad, r2_test_grad],
    'Ridge': [rmse_train_ridge, rmse_test_ridge, mse_train_ridge, mse_test_ridge, r2_train_ridge, r2_test_ridge],
    'Cross Validation 5': [mean_rmse_train5, mean_rmse_test5, mean_mse_train5, mean_mse_test5, mean_r2_train5, mean_r2_test5],
    'Stochstic gradient descent': [rmse_train_SDG, rmse_test_SDG, mse_train_SDG, mse_test_SDG, r2_train_SDG, r2_test_SDG],
    'Mini Batch Gradient Descent': [rmse_train_mini, rmse_test_mini, mse_train_mini, mse_test_mini, r2_train_mini, r2_test_mini]
}



modelalter_df=pd.DataFrame(data,index=['rmse-train', 'rmse-test', 'mse-train', 'mse-test', 'r2-train', 'r2-test'])

modelalter_df = modelalter_df.applymap(lambda x: f"{x:.6f}")
print(modelalter_df)

# Для кросс-валидации можно увидеть гораздо большие значения ошибок, однако кросс-валидация обладает наибольшим значением параметра r^2,
# Наихудшим образом себя показала модель gradient_Descent, обладает высокими значениями ошибок с самыми низкими значениями r^2 на обеих выборках, при чем на тестовой r^2 в два раза ниже чем на тренировочной
# Stochstic gradient descent и Mini Batch Gradient Descent, показали примерно похожие результаты, однако модель Mini Batch Gradient Descent оказалась все же более показательной 
# Результаты модели ridge очень близки к Mini Batch Gradient Descent. Из всего вышеперечисленного, стоит сказать, что наибболее адекватной моделью по ряду параметров можно назвать ridge 


