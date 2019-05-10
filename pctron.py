import numpy as np
import cv2
from sklearn.utils import shuffle
class Perceptron():
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.dims = [2,128,2]
        self.w1 = 0.01*np.random.randn(self.dims[0],self.dims[1])
        self.b1 = np.zeros((self.dims[1],1))
        self.w2 = 0.01*np.random.randn(self.dims[1],self.dims[2])
        self.b2 = np.zeros((self.dims[2],1))
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def propaganate(self,X,Y,w,b):
        A = self.sigmoid(np.dot(w.T,X) +b)
        m = X.shape[1]
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A-Y)/m
        cost = (-1  / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return dw,db,cost
    def cost(self,Y, Yhat):
        return -np.sum(Y*np.log(Yhat))/Y.shape[1]
    def optimize(self,learningRate=0.005,steps=500):
        X = self.x_train
        Y = self.y_train
        w1 = self.w1
        b1 = self.b1
        w2 = self.w2
        b2 = self.b2
        N = X.shape[1]
        costs =[]
        for i in range(steps):
            z1 = np.dot(w1.T,X.T)+b1
            a1 = np.maximum(z1,0)
            z2 = np.dot(w2.T,a1)+b2
            y_h = self.sigmoid(z2)
            if i%100 ==0:
                cost = self.cost(Y,y_h)
                costs.append(cost)
                print('cost after %i: %f' %(i,cost))
            e2 = (y_h-Y)/N
            dw2 = np.dot(a1,e2.T)
            db2 = np.sum(e2,axis=1,keepdims= True)
            e1 = np.dot(w2,e2)
            e1[z1<=0] =0
            dw1 = np.dot(X.T,e1.T)
            db1 = np.sum(e1,axis=1,keepdims= True)
            w1 += -learningRate*dw1
            b1 += -learningRate*db1
            w2 += -learningRate*dw2
            b2 += -learningRate*db2
        print('bat dau save ...')
        self.save(w1,b1,w2,b2)
        print('saved')
        return w1,b1,w2,b2
    def load(self):
        w1 = np.load('weights1.npy')
        b1 = np.load('biases1.npy')
        w2 = np.load('weights2.npy')
        b2 = np.load('biases2.npy')
        return w1,b1,w2,b2
    def save(self,w1,b1,w2,b2):
        np.save('weights1.npy',w1)
        np.save('biases1.npy',b1)
        np.save('weights2.npy',w2)
        np.save('biases2.npy',b2)
    def predict(self,image):
        w1,b1,w2,b2 = self.load()
        m = image.shape[1]
        Z1 = np.dot(w1.T, image) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(w2.T, A1) + b2
        A2 = self.sigmoid(z2)
        for i in range(A.shape[1]):
            Y_prediction[0,i] =A[0,i]
        print(Y_prediction)
        return Y_prediction
    

x_train,y_train = [],[]
for i in range(1):
    img = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\cat.' + str(i) + '.jpg')
    img = cv2.resize(img,(64,64))
    x_train.append(img)
    y_train.append(0)
for i in range(1):
    img = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\dog.' + str(i) + '.jpg')
    img = cv2.resize(img,(64,64))
    x_train.append(img)
    y_train.append(1)
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
x_train, y_train = shuffle(x_train, y_train, random_state = 0)
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
x_train = x_train_flatten / 255
pct = Perceptron(x_train,y_train)
# pct.optimize()
pd_img = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\dog.0.jpg')
pd_img = cv2.resize(pd_img,(64,64))
pd_img = np.array(pd_img)
predict_imgs_flatten = pd_img.reshape(pd_img.shape[0],-1).T
pd_img = predict_imgs_flatten/255
print(pd_img.shape)
pct.predict(pd_img)
