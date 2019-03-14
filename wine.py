#coding:utf-8
from __future__ import division
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
################定义激活函数~##########################################

def sigmoid_activation(x, theta):
    x = np.asarray(x)
    theta = np.asarray(theta)
    return 1 / (1 + np.exp(-np.dot(theta.T, x)))



# 将模型的函数凝结为一个类，这是很好的一种编程习惯
class NNet3:
    # 初始化必要的几个参数
    def __init__(self, learning_rate=0.5, maxepochs=1e4, convergence_thres=1e-5, hidden_layer=4):
        self.learning_rate = learning_rate
        self.maxepochs = int(maxepochs)
        self.convergence_thres = 1e-5
        self.hidden_layer = int(hidden_layer)

    #  计算最终的误差
    def _multiplecost(self, X, y):
        # l1是中间层的输出，l2是输出层的结果
        l1, l2 = self._feedforward(X)
        # 计算误差，这里的l2是前面的h
        inner = y * np.log(l2) + (1-y) * np.log(1-l2)
        # 添加符号，将其转换为正值
        return -np.mean(inner)

    # 前向传播函数计算每层的输出结果
    def _feedforward(self, X):
        # l1是中间层的输出
        l1 = sigmoid_activation(X.T, self.theta0).T
        # 为中间层添加一个常数列
        l1 = np.column_stack([np.ones(l1.shape[0]), l1])
        # 中间层的输出作为输出层的输入产生结果l2
        l2 = sigmoid_activation(l1.T, self.theta1)
        return l1, l2

    # 传入一个结果未知的样本，返回其属于1的概率
    def predict(self, X):
        _, y = self._feedforward(X)
        return y

    # 学习参数，不断迭代至参数收敛，误差最小化
    def learn(self, X, y):
        nobs, ncols = X.shape
        self.theta0 = np.random.normal(0,0.01,size=(ncols,self.hidden_layer))
        self.theta1 = np.random.normal(0,0.01,size=(self.hidden_layer+1,1))

        self.costs = []
        cost = self._multiplecost(X, y)
        self.costs.append(cost)
        costprev = cost + self.convergence_thres+1
        counter = 0

        for counter in range(self.maxepochs):
            # 计算中间层和输出层的输出
            l1, l2 = self._feedforward(X)

            # 首先计算输出层的梯度，再计算中间层的梯度
            l2_delta = (y-l2) * l2 * (1-l2)
            l1_delta = l2_delta.T.dot(self.theta1.T) * l1 * (1-l1)

            # 更新参数
            self.theta1 += l1.T.dot(l2_delta.T) / nobs * self.learning_rate
            self.theta0 += X.T.dot(l1_delta)[:,1:] / nobs * self.learning_rate

            counter += 1
            costprev = cost
            cost = self._multiplecost(X, y)  # get next cost
            self.costs.append(cost)
            if np.abs(costprev-cost) < self.convergence_thres and counter > 500:
                break



if __name__ == '__main__':

    ###################################################数据处理#############################
    #############读取数据##################
    train_data = pandas.read_csv("./WineQualityTrain.csv", encoding='utf-8', low_memory=False, na_values='\\N').fillna(0)

    # 打乱数据顺序################
    shuffled_rows = np.random.permutation(train_data.index)

    train_data = train_data.iloc[shuffled_rows]

    # 添加一个值全为1的属性train_data["ones"]，截距
    train_data["ones"] = np.ones(train_data.shape[0])
    X = train_data[['ones', 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']].values

    # 将train_data-versicolor类标签设置为1，train_data-virginica设置为0
    y = train_data.type.values

    # First 70 rows to X_train and y_train
    # Last 30 rows to X_train and y_train
    X_train = X[:2498]
    y_train = y[:2498]

    X_test = X[-1070:]
    y_test = y[-1070:]


    ###################################################训练模型##################################


    # Set a learning rate
    learning_rate = 0.8
    # Maximum number of iterations for gradient descent
    maxepochs = 10000
    # Costs convergence threshold, ie. (prevcost - cost) > convergence_thres
    convergence_thres = 0.00001
    # Number of hidden units
    hidden_units = 8

    # Initialize model
    model = NNet3(learning_rate=learning_rate, maxepochs=maxepochs,
                  convergence_thres=convergence_thres, hidden_layer=hidden_units)
    model.learn(X_train, y_train)
    train_yhat = model.predict(X_train)[0]

    print (y_train)
    print (train_yhat)
    train_auc = roc_auc_score(y_train, train_yhat)

    print (train_auc)



    ########################################预测数据##############################

    # 因为predict返回的是一个二维数组，此处是(1,30)，取第一列作为一个列向量
    yhat = model.predict(X_test)[0]

    print (y_test)
    print (yhat)






    predict=[]
    for each in yhat:
        if each>0.5:
            predict.append(1)
        else:
            predict.append(0)


    print (predict)

    auc = roc_auc_score(y_test, yhat)

    print (auc)


    #################################写出数据##########################
    ######################合并各列数据#########################
    result=np.column_stack([X_test,y_test,predict])
    print (result)
    count=0
    for i in range(0,len(result)):
         if result[i,12]==result[i,13]:
             count+=1

    ################计算准确率#############################
    print (count,len(result))
    acurate=count/len(result)


    print("分类正确率是：%.2f%%" % (acurate * 100))

    labels = list(set(predict))
    print (labels)
    conf_mat = confusion_matrix(y_test, predict, labels=labels)

    print (conf_mat)




    #####################数组转换为数据框##########################
    result=pandas.DataFrame(result[:,1:])

    result.columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol', 'type', 'predict']


    print (result)
    #########################写出数据到excel################

    pandas.DataFrame.to_excel(result,"./train_data_test.xlsx",index=False)



    # Plot costs
    plt.plot(model.costs,color="red")
    plt.title("Convergence of the Cost Function")
    plt.ylabel("J($\Theta$)")
    plt.xlabel("Iteration")
    plt.show()
