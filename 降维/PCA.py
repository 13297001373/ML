import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
##创造数据
def load_data(path):
    data = []
    with open(path,'r') as f:
        for line in f.readlines():
            row = line.strip().split(',')
            row = list(map(lambda x:float(x),row))
            data.append(row)
    data = np.array(data)
    return data[:,0],data[:,1:]
def plot_data(X,y):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y)
    plt.title('主成分分析PCA')
    plt.show()
class PCA():
    def __init__(self,k):
        self.eigen_values = None
        self.eigen_vectors = None
        self.k = k
    def pca_transform(self,X):
        ##1.数据标准化
        mean = np.mean(X, axis=0, keepdims=True)
        X = X-mean
        ##2.计算标准化矩阵的协方差矩阵
        n = X.shape[0]
        cov_X = X.T.dot(X)/n
        ##3.计算协方差矩阵的特征值和特征向量
        self.eigen_values,self.eigen_vectors = np.linalg.eig(cov_X)
        idx = self.eigen_values.argsort()[::-1]
        eigen_values = self.eigen_values[idx][:self.k]
        eigen_vectors = self.eigen_vectors[:,idx][:,:self.k]
        return X.dot(eigen_vectors)
def test():
    ##1.读取数据
    path = './data/wine.data'
    y,X = load_data(path)
    ##2.数据降维
    pca = PCA(3)
    X = pca.pca_transform(X)
    ##3.绘制图形
    plot_data(X,y)
if __name__ == '__main__':
    test()