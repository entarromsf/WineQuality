import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn import svm
n_splits = 5
random_seed = 67
lgb_params = {
    # 调参主要调这五个
    "learning_rate":0.1,
    "lambda_l1":0.1,
    "lambda_l2":0.2,
    "max_depth":6,
    "num_leaves":50,
    # 这几个固定不动就可以
    "application":"multiclass", # 问题类型
    "num_class":11, # 分类数量
    "num_thread":4, # 线程数 推荐改为CPU物理核数
    "verbose":-1, # 静默模式
}

def SVM(X, y, X_sub):
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)
    clf.fit(X, y)
    sub_test = pd.DataFrame()
    sub_test['id'] = clf.predict(X_sub)
    return sub_test

# 自定义评价函数 为宏平均的平方
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1), axis=0)
    score_vali = np.square(f1_score(y_true=labels, y_pred=preds, average='macro'))
    return 'f1_macro^2', score_vali, True

def lgbm(X, y, X_sub) :
    xx_score = []
    cv_pred = []

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)

    for index, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print("CV Training:", index)
        X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]

        train_data_tmp = lgb.Dataset(X_train, label=y_train)
        valid_data_tmp = lgb.Dataset(X_valid, label=y_valid)

        clf = lgb.train(lgb_params, train_data_tmp, num_boost_round=10000, valid_sets=[valid_data_tmp],
                        early_stopping_rounds=50, feval=f1_score_vali, verbose_eval=1)

        y_valid_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)

        y_valid_pred = [np.argmax(x) for x in y_valid_pred]

        xx_score.append(np.square(f1_score(y_valid, y_valid_pred, average='macro')))

        y_sub = clf.predict(X_sub, num_iteration=clf.best_iteration)

        y_sub = [np.argmax(x) for x in y_sub]

        if index == 0:
            cv_pred = np.array(y_sub).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_sub).reshape(-1, 1)))

    # 投票
    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))

    # 保存结果
    sub_test = pd.DataFrame()
    # sub_test['id'] = list(sub_data_id.unique())
    sub_test['id'] = submit
    # sub_test['predict'] = sub_test['predict'].map(label2current_service)
    return sub_test
    print(xx_score, np.mean(xx_score))

def main():
    train_data = pd.read_csv("./WineQualityTrain.csv", encoding='utf-8', low_memory=False, na_values='\\N').fillna(0)
    sub_data = pd.read_csv("./WineQualityTest.csv", encoding='utf-8', low_memory=False, na_values='\\N').fillna(0)

    # 分离feature和lable
    train_data_y = train_data.pop('type')
    train_data_x = train_data
    sub_data_x = sub_data

    sub_data_x = sub_data_x[train_data_x.columns]
    columns = train_data_x.columns.tolist()
    X, y, X_sub = train_data_x.values, train_data_y.values, sub_data_x.values
    model_type = input()
    if(model_type == 'lgbm') :
        sub_test = lgbm(X, y, X_sub)
        sub_test.to_csv('./result/lgbm.csv', index=False)
    elif(model_type == 'svm'):
        sub_test = SVM(X, y, X_sub)
        sub_test.to_csv('./result/svm.csv', index=False)
    else:
        print("ERROR")

if __name__ == "__main__":
    main()
