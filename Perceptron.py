
# coding: utf-8

# In[11]:

import numpy as np
class Perceptron(object):
    """
    eta :学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """
    def __int__(self,eta = 0.01, n_iter = 10):
        self.eta = eta;
        self.n_iter = n_iter
        pass
    def fit(self,X,y):
        """
        输入训练数据，培训神经元，X输入样本向量，y对应的样本分类
        X:shap[n_samples,n_features]
        X:[[1,2,3],[4,5,6]]
        n_samples:2
        n_features:3
        
        y:[1,-1]
        """
        """
        初始化权重向量为0
        加一是因为前面算法所提到的w0，也就是步调函数的阈值
        """
        self.w_ = np.zeros(1 + X.shape[1]);
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            """
             X:[[1,2,3],[4,5,6]]
             y:[1,-1]
             zip(X,y) = [[1,2,3, 1],[4,5,6, -1]]
             xi:[1,2,3],...
             target:1,...
            """
            for xi,target in zip(X,y):
                """
                update = η *（y-y′）
                predict：调用的函数
                """
                update = self.eta *(target-self.predict(xi))
                """
                xi是一个向量
                update * xi等价：
                [▼W(1) = X[1]*update,▼W(2) = X[2]*update,▼W(3) = X[3]*update,
                self.w[1:]:
                从w数组的第二个元素开始更新，w[0]为阈值
                """
                self.w[1:] += update * xi  
                self.w[0] +=update;
                errors += int(update!=0.0)
                self.errors_.append(errors)
                
                pass
                """
                对电信号和权重进行点积并且加上阈值
                """
                def net_input(self, X):
                    """
                    np.dot():
                    z = W0*1 + W1 * x1 +...Wn *Xn
                    """
                    return np.dot(X,self.w_[1:]) + self.w_[0]
                    pass
                """
                判断电信号X的的分类
                """
                def predict(self,X):
                    return np.where(self.net_input(X) >= 0.0 ,1,-1)
                    pass
                pass


    


# In[6]:

file = 'F:\iris.csv'
#数据读取类库
import pandas as pd
df = pd.read_csv(file, header=None)
#df.head(10)


# In[7]:

#数据可视化工具
import matplotlib.pyplot as plt
#对数据进行处理加工
import numpy as np
#解决乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.serif'] = ['SimHei']
y = df.loc[0:100,4].values
#print(y)
y = np.where(y == 'Iris-setosa',-1,1)

X = df.iloc[0:100,[0, 2]].values 
#print(X)
#将前50条和50到100条通过坐标轴画出来
plt.scatter(X[:50, 0],X[:50, 1],color='red',marker='o', label='setosa')
plt.scatter(X[50:100,0],X[50:100, 1],color='blue',marker='x', label='versicolor')
plt.xlabel('花瓣长度')
plt.ylabel('花茎长度')
plt.legend(loc='upper left')
plt.show()


# In[40]:


ppn = Perceptron()
"""
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlable('Epochs')
plt.ylabel('错误分类次数')
"""


# In[44]:

from matplotlib.colors import ListedColormap
def plot_decision_regions(X,y,classifier, resolution = 0.02):#定义实现对数据进行分类的函数
    marker = ('s','x','o','v')#数据展示时的信息
    colors = ('red','blue','lightgreen','gray','cyan')#颜色列表
    cmap = ListedColormap(colors[:len(np.unique(y))])#根据不同种类的y值分配不同的颜色
    #统计花茎和花瓣的最大值和最小值
    x1_min,x1_max = X[:, 0].min() - 1,X[:, 0].max()
    x2_min,x2_max = X[:, 1].min() - 1,X[:, 1].max()
    
    print(x1_min,x1_max)
    print(x2_min,x2_max)
    #将两个向量扩展为变成矩阵
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                            np.arange(x2_min,x2_max,resolution))
    
    #print(np.arange(x1_min,x1_max,resolution).shape)
    #print(np.arange(x1_min,x1_max, resolution))
    #print(xx1.shape)
    #print(xx1)
    #print(xx2.shape)
    #print(xx2)
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(xx1.ravel())
    print(xx2.ravel())
    print(Z)
    
    Z = Z.reshape(xx1,shape)#将Z转化成和xx1一样的二维数组
    plt.contourf(xx1,xx2,Z,alpha = 0.4, cmap = cmap)#根据所给的xx1，xx2绘制出分界线
    #根据起始节点和末尾节点对应上坐标的最小值和最大值
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    #给相应的数据节点打上相应的数据说明
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y = X[y==cl, 1], alpha = 0.8,c = cmap(idx),
                   marker=markers[idx], label=cl)
        
    


# In[45]:

plot_decision_regions(X,y,ppn,resolution = 0.02)
plt.xlabel('花茎长度')
plt.ylabel('花瓣长度')
plt.legend(loc = 'upper left')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



