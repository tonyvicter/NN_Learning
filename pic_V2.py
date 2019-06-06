#from PIL import Image
import numpy as np

def preprocessing(listOfexamples): #导入图片
    '''
    xxx
    '''


def initial_weight(n): #初始化权重W, b
    '''
    W = [n*n*3,1]
    b = [1,1]
    return W,b
    '''

def forward_backward_prop(W, b, X, Y): #定义神经网络模型的函数
    '''
    Z
    A
    J
    dZ
    dW
    db
    alpha = 0.5
    W:= W - alpha*dW
    b:= b - alpha*db
    return W, b
    '''

# main()
# loop
def main():
    '''
    import matplotlib.pyplot as plt
    jseries = []
    X, Y = preprocessing(listOfexamples)
    W, b = initial_weight(n)
    for i in range(200):
        W, b, J= forward_backward_prop(W, b, X, Y)
        jseries.append(J)
    '''
    return None

'''
plt.subplot(1,1,1)
plt.plot(jseries)
plt.show()
print(w)
print(b)
'''

# predict
'''
pic -> x array -> np.dot(w.T, x)+b
90% for training
10% for testing
'''
