import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def load_data(): # 导入训练样本，本利为图片
    import os
    from PIL import Image # 放在此处引入，与放在最上面有什么区别？
    X = np.array([]) # 初始化输入矩阵x
    for root, dir, files in os.walk(os.path.join(os.getcwd(), "pics_cat")):
    #for root, dir, files in os.walk(os.path.join(os.getcwd(), "NN_Learning/pics_cat")):
        for sfile in files:
            try:
                print(root)
                print(dir)
                print(sfile)

                # 读取图片，resize图片，获得三个通道的数据
                im = Image.open(os.path.join(root, sfile))
                im = im.resize((30, 30))
                im_arr = np.array(im)
                arr_r = im_arr[:, :, 0]
                arr_g = im_arr[:, :, 1]
                arr_b = im_arr[:, :, 2]

                # hstack是numpy的内建函数，用于水平堆叠序列中的数组（列方向），与np.concatenate((a,b,c),axis=0)的作用类似
                # arr.flat，按照从左至右，从上至下的顺序，依次列出数组中的所有元素
                pixial_array = np.hstack((arr_r.flat, arr_g.flat, arr_b.flat))
                # (-1,1)表示列数确定为1，行数自动计算。反过来（1，-1）表示行数确定为1，行数自动计算。
                pixial_array = np.reshape(pixial_array, (-1, 1))

                if X.shape[0] == 0:
                    X = pixial_array
                else:
                    X = np.hstack((X, pixial_array))

            except:
                continue
    print(X)
    print(X.shape)
    X = X/255
    m = X.shape[1] # number of examples, X.shape = (c, l)，X.shape[1] = l
    Y = np.ones(shape=(1,m))
    return X, Y

def iniParams(X): #初始化权重W, b
    print(X.shape)
    W = np.random.rand(X.shape[0], 1)/X.shape[0]
    import random
    b = random.random()
    return W, b
    '''
    W = [n*n*3,1]
    b = [1,1]
    return W,b
    '''

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def f_b_prop(W, b, X, Y): #定义神经网络模型的函数
    m = X.shape[1] # X.shape[1]表示矩阵X的列数，即，样本个数
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    J = (np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A)))) / m
    dZ = A - Y
    dW = np.dot(X, dZ.T) / m #除以m
    db = np.sum(dZ) / m #除以m
    alpha = 0.01
    W -= alpha * dW
    b -= alpha * db
    return W, b, J

# main()
# loop
def main():
    X, Y = load_data() #导入样本
    print(X)
    print(Y)

    shape_X = X.shape #获得矩阵X的纬度，(c, l)
    shape_Y = Y.shape #获得矩阵Y的纬度，(c, l)
    n = X.shape[1] #获得样本个数
    print()
    print("------------")
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('We have n = ' + str(n) + ' traning examples')

    #初始化w和b
    W, b = iniParams(X)
    print()
    print("the random weight and bias would be ")
    print("W: ", iniParams(X)[0])
    print("b: ", iniParams(X)[1])

    #创建用于绘图的容器
    plotJ = []  # 保存J的空list
    plotw1 = [] # 保存？的空list
    plotw2 = [] # 保存？的空list
    plotb = []  # 保存b的空list

    #训练过程
    iter = 0
    while iter < 300:
        W, b, J = f_b_prop(W, b, X, Y)
        plotJ.append(J)         # 保存J的结果，在末尾添加新的变量
        plotw1.append(W[0, 0])  # 保存？的结果，在末尾添加新的变量
        plotw2.append(W[1, 0])  # 保存？的结果，在末尾添加新的变量
        plotb.append(b)         # 保存b的结果，在末尾添加新的变量
        print(str(iter)+"-th iteration, J: " + str(J))  #显示当前进行的是第几次计算，并列出本次计算得到的J
        iter += 1
    
    print()
    print("Optimized W: ", W)
    print("Optimized b: ", b)
    
    plt.subplot(4, 1, 1)
    plt.plot(plotJ)
    plt.subplot(4, 1, 2)
    plt.plot(plotw1)
    plt.subplot(4, 1, 3)
    plt.plot(plotw2)
    plt.subplot(4, 1, 4)
    plt.plot(plotb)
    plt.show()

# predict
'''
pic -> x array -> np.dot(w.T, x)+b
90% for training
10% for testing
'''
