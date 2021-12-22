import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

#read data
dataset = pd.read_csv('raw_dataset.csv',engine='python')
#set variable
rs = np.random.RandomState(169)
outliers_fraction = 0.05
lendata = dataset.shape[0]
#label
anomaly = [];test_data = []
#normalize
nmlz_a = -1 
nmlz_b = 1

def normalize(dataset,a,b):
    scaler = MinMaxScaler(feature_range=(a, b))
    normalize_data = scaler.fit_transform(dataset)
    return normalize_data

#read dataset x,y
x = normalize(pd.DataFrame(dataset, columns=['cr']), nmlz_a, nmlz_b)
y = normalize(pd.DataFrame(dataset, columns=['wr']), nmlz_a, nmlz_b)
# print(x)

ifm = IsolationForest(n_estimators=100, verbose=2, n_jobs=2,
                      max_samples=lendata, random_state=rs, max_features=2)

if __name__ == '__main__':

    Iso_train_dt = np.column_stack((x, y))
    ifm.fit(Iso_train_dt)
    scores_pred = ifm.decision_function(Iso_train_dt)
    print(scores_pred)
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    print(threshold)
    xx, yy = np.meshgrid(np.linspace(nmlz_a, nmlz_b, 50), np.linspace(nmlz_a, nmlz_b, 50))
    Z = ifm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest ")
    otl_proportion = int(outliers_fraction * len(dataset['Date']))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, otl_proportion), cmap=plt.cm.hot)# 绘制异常点区域，值从最小的到阈值的那部分
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')# 绘制异常点区域和正常点区域的边界
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='palevioletred')
    # 绘制正常点区域，值从阈值到最大的那部分

    for i in scores_pred:
        if i <= threshold:
            #print(i)
            test_data.append(1)
            anomaly.append(i)
        else:
            test_data.append(0)

    ano_lable = np.column_stack(((dataset['Date'],dataset['data'],x,y,scores_pred, test_data)))
    df = pd.DataFrame(data=ano_lable, columns=['Date','data','x', 'y', 'IsoFst_Score','label'])

    b = plt.scatter(df['x'][df['label'] == 0], df['y'][df['label'] == 0], s=20, edgecolor='k',c='white')
    c = plt.scatter(df['x'][df['label'] == 1], df['y'][df['label'] == 1], s=20, edgecolor='k',c='black')
    plotlist = df.to_csv('processed_data.csv')

    plt.axis('tight')
    plt.xlim((nmlz_a, nmlz_b))
    plt.ylim((nmlz_a, nmlz_b))
    plt.legend([a.collections[0], b, c],
               ['learned decision function', 'true inliers', 'true outliers'],
               loc="upper left")
    print("孤立森林阈值  ：",threshold)
    print("全量数据样本数：",len(dataset),"个")
    print("检测异常样本数：",len(anomaly),"个")
    # print(anomaly)
    plt.show()

new_dataset = pd.read_csv('processed_data.csv',engine='python')

plt.figure(1)
plt.subplot(2,1,1)
plt.title('iForest')
plt.plot(new_dataset.data)
for i in range(len(new_dataset)):
    if(new_dataset.label[i] == 1):
        plt.scatter(i, new_dataset.data[i], c='r')

plt.subplot(2,1,2)
plt.title('dictionary')
plt.plot(new_dataset.data)
# 字典法，设置阈值为 0.04 和 0.042
for i in range(len(dataset)):
    if(abs(dataset.cr[i]) > 0.040 or abs(dataset.wr[i]) > 0.042):
        plt.scatter(i, dataset.data[i], c='r')

# plt.figure(1)
# plt.subplot(2,1,1)
# plt.title('cr')
# plt.plot(dataset.cr)
# for i in range(len(new_dataset)):
#     if(new_dataset.label[i] == 1):
#         plt.scatter(i, dataset.cr[i], c='r')

# plt.subplot(2,1,2)
# plt.title('wr')
# plt.plot(dataset.wr)
# for i in range(len(new_dataset)):
#     if(new_dataset.label[i] == 1):
#         plt.scatter(i, dataset.wr[i], c='r')

plt.show()
