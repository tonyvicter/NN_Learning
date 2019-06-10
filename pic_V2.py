from PIL import Image
import numpy as np

def preprocessing(listOfexamples): #导入图片
    im = Image.open(listOfexamples)
    r = im.red
    b = im.blue
    g = im.green
    X = np.concatenate(r, b ,g)
    Y = [1]
    return X, Y

def initial_weight(n): #初始化权重W, b
    
    W = [n*n*3,1]
    b = [1,1]
    return W,b

def forward_backward_prop(W, b, X, Y): #定义神经网络模型的函数
    m = len(X)
    Z = np.dot(W.T, X) + b
    A = 1 / (1 + np.exp(-Z))
    J = (np.sum(-(Z * np.log(A) + (1 - Z) * np.log(1 - A)))) / m
    dZ = A - Y
    dW = np.dot(X, dZ.T)
    db = np.sum(dZ) / m
    alpha = 0.5
    W = W - alpha * dW
    b = b - alpha * db
    return W, b, J

# main()
# loop
def main():
    # import matplotlib.pyplot as plt
    jseries = [] #初始化一个空的list
    listOfexamples = input('输入图片地址：')
    X, Y = preprocessing(listOfexamples)
    n = len(X(0))
    W, b = initial_weight(n)
    
    iter = 0
    while iter < 200:
        W, b, J= forward_backward_prop(W, b, X, Y)
        jseries.append(J) #在末尾添加新的变量
        iter += 1
    return None
    
    # plt.subplot(1,1,1)
    # plt.plot(jseries)
    # plt.show()
    # print(W)
    # print(b)

# predict
'''
pic -> x array -> np.dot(w.T, x)+b
90% for training
10% for testing
'''
