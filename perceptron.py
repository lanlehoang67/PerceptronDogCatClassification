import numpy as np
import cv2
from sklearn.utils import shuffle
class Perceptron():
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.w = np.zeros((self.x_train.shape[0],1))
        self.b =0
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def propaganate(self,X,Y,w,b):
        A = self.sigmoid(np.dot(w.T,X) +b)
        m = X.shape[1]
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A-Y)/m
        cost = (-1  / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return dw,db,cost
    def optimize(self,learningRate=0.005,steps=2000):
        X = self.x_train
        Y = self.y_train
        w = self.w
        b = self.b
        costs =[]
        for i in range(steps):
            dw,db,cost =self.propaganate(X,Y,w,b)
            w = w - learningRate*dw
            b = b - learningRate*db
            if i%100 ==0:
                costs.append(cost)
                print('cost after %i: %f' %(i,cost))
        print('bat dau save ...')
        self.save(w,b)
        print('saved')
        return w,b,dw,db,costs
    def load(self):
        w = np.load('weights.npy')
        b = np.load('biases.npy')
        return w,b
    def save(self,w,b):
        np.save('weights.npy',w)
        np.save('biases.npy',b)
    def predict(self,image):
        w,b = self.load()
        m = image.shape[1]
        w = w.reshape((image.shape[0],-1))
        Y_prediction = np.zeros((1,m))
        A = self.sigmoid(np.dot(w.T,image)+b)
        for i in range(A.shape[1]):
            Y_prediction[0,i] =A[0,i]
        print(Y_prediction)
        return Y_prediction
    

x_train,y_train = [],[]
for i in range(1):
    img = cv2.imread('E:\\dog-or-cat\\dataset\\train\\training\\cat.' + str(i) + '.jpg')
    img = cv2.resize(img,(64,64))
    x_train.append(img)
    y_train.append(0)
for i in range(1):
    img = cv2.imread('E:\\dog-or-cat\\dataset\\train\\training\\dog.' + str(i) + '.jpg')
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
pd_img = cv2.imread('E:\\dog-or-cat\\dataset\\train\\training\\dog.0.jpg')
pd_img = cv2.resize(pd_img,(64,64))
pd_img = np.array(pd_img)
predict_imgs_flatten = pd_img.reshape(pd_img.shape[0],-1).T
pd_img = predict_imgs_flatten/255
pct.predict(pd_img)
