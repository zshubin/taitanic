# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import model_selection
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("./Train.csv")
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
# plt.title(u"survive status (1:survivor)")  # 标题
# plt.ylabel(u"number")
#
# plt.subplot2grid((2, 3), (0, 1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"number")
#
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age, marker='.')
# plt.ylabel(u"age")  # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"split by age (1:survivor)")
#
# plt.subplot2grid((2, 3), (1, 0))
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"age")  # plots an axis lable
# plt.ylabel(u"density")
# plt.legend((u'first level', u'second level', u'third level'), loc='best')  # sets our legend for our graph.
#
# plt.subplot2grid((2, 3), (1, 1))
# data_train.Age[data_train.Survived == 0].plot(kind='kde')
# data_train.Age[data_train.Survived == 1].plot(kind='kde')
# plt.xlabel(u"age")  # plots an axis lable
# plt.ylabel(u"density")
# plt.legend((u'dead', u'survived'), loc='best')
#
# plt.subplot2grid((2, 3), (1, 2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"booting number of every port")
# plt.ylabel(u"number")
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.2)
#
# Survived_0 = data_train.Age[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Age[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'survive': Survived_1, u'dead': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.xlabel(u"number")  # plots an axis lable
# plt.ylabel(u"density")
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'survive': Survived_1, u'dead': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.xlabel(u"class")
# plt.ylabel(u"number")
# plt.show()
#
# # 看看各性别的获救情况
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df = pd.DataFrame({u'male': Survived_m, u'famale': Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.xlabel(u"sex")
# plt.ylabel(u"number")
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'survive': Survived_1, u'dead': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.xlabel(u"port")
# plt.ylabel(u"number")
# plt.show()

# g = data_train.groupby(['SibSp', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
# g = data_train.groupby(['SibSp', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
#
# data_train.Cabin.value_counts()

def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df[['Age']])
# fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])
#
# # 对cross validation数据进行预测
#
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:, 1:])
#
# origin_data_train = pd.read_csv("./train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
# print(bad_cases)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


plot_learning_curve(clf, u"学习曲线", X, y)

# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.as_matrix()[:,1:]
# y = all_data.as_matrix()[:,0]
# print(model_selection.cross_val_score(clf, X, y, cv=5))
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(X, y)

# data_test = pd.read_csv("./test.csv")
# data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
# # 接着我们对test_data做和train_data中一致的特征变换
# # 首先用同样的RandomForestRegressor模型填上丢失的年龄
# tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
# null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# # 根据特征属性X预测年龄并补上
# X = null_age[:, 1:]
# predictedAges = rfr.predict(X)
# data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges
#
# data_test = set_Cabin_type(data_test)
# dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
# dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
# dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
# dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
#
# df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
# df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']])
# df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']])
#
# test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
# result.to_csv("./logistic_regression_predictions.csv", index=False)

# f = pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
# print('a')