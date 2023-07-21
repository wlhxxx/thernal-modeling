from Data_Standard_Multi import data_prepard, data_labels, data_prepard_1, data_labels_1
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import datetime

SVR_model = SVR(C=150)
regr = MultiOutputRegressor(SVR_model)


# regr.fit(data_prepard, data_labels)
# regr.predict(data_prepard)
# print(regr.predict(data_prepard))

# 定义模型结果函数
def model_score(model, x, y):
    cv = ShuffleSplit(n_splits=10, train_size=0.8, test_size=0.2, random_state=29)

    rmse = cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=cv)
    rmse_score = np.sqrt(-rmse)
    rmse_mean = rmse_score.mean()
    rmse_std = rmse_score.std(ddof=1)

    mae = cross_val_score(model, x, y, scoring="neg_mean_absolute_error", cv=cv)
    mae_score = -mae
    mae_mean = mae_score.mean()
    mae_std = mae_score.std(ddof=1)

    r2 = cross_val_score(model, x, y, scoring='r2', cv=cv)
    r2_mean = r2.mean()
    r2_std = r2.std(ddof=1)  # 无偏估计

    model_scores = [rmse_score, rmse_mean, rmse_std, r2, r2_mean, r2_std, mae_score, mae_mean, mae_std]
    return model_scores


starttime = datetime.datetime.now()
SVR_score = model_score(regr, data_prepard, data_labels)
endtime = datetime.datetime.now()
print(endtime - starttime)

writer = pd.ExcelWriter('Multi-SVR_score.xlsx')

model_scores = SVR_score
rmse_score = pd.DataFrame(model_scores[0])
rmse_mean = model_scores[1]
rmse_std = model_scores[2]
r2 = pd.DataFrame(model_scores[3])
r2_mean = model_scores[4]
r2_std = model_scores[5]
mae_score = pd.DataFrame(model_scores[6])
mae_mean = model_scores[7]
mae_std = model_scores[8]
result = [rmse_mean, rmse_std, r2_mean, r2_std, mae_mean, mae_std]
result_name = pd.DataFrame(['rmse_mean', 'rmse_std', 'r2_mean', 'r2_std', 'mae_mean', 'mae_std'])
result_pd = pd.DataFrame(np.array(result))

model_scores_pd = pd.concat([rmse_score, r2, mae_score, result_pd, result_name], axis=1)
model_scores_pd.columns = ['RMSE', 'R2', 'MAE', 'Results', 'Results_name']

model_scores_pd.to_excel(writer, sheet_name='SVR')
writer.save()
