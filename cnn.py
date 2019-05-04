import skimage.data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ConvolutionalNeuralNetwork():
    def __init__(self,x_train,y_train,l1_filter):
        self.x_train = x_train
        self.y_train = y_train
        self.l1_filter = l1_filter
    def sigmoid(self,input):
        return 1/(1+np.exp(-input))
    def sigmoid_prime(self,input):
        s = 1/(1+np.exp(-input))
        return np.dot(s,1-s)
    def conv(self,x_train,conv_filter):
        if len(x_train.shape) > 2 or len(conv_filter.shape) > 3: 
            if x_train.shape[-1] != conv_filter.shape[-1]:
                print("so chieu trong anh va filter phai trung")             
                sys.exit()      
        if conv_filter.shape[1] != conv_filter.shape[2]: 
            print('phai la ma tran vuong, hang cot phai nhu nhau')  
            sys.exit()  
        if conv_filter.shape[1]%2==0: 
            print('filter phai la so chan')  
            sys.exit()  
        feature_maps = np.zeros((x_train.shape[0]-conv_filter.shape[1]+1,x_train.shape[1]-conv_filter.shape[1]+1,conv_filter.shape[0]))
        for filter_num in range(conv_filter.shape[0]):
            print("filter",filter_num+1)
            curr_filter = conv_filter[filter_num,:]
            if len(curr_filter.shape) >2:
                conv_map = self.conv_(x_train[:,:,0],curr_filter[:,:,0])
                for ch_num in range(1,curr_filter.shape[-1]):
                    conv_map = conv_map + self.conv_(x_train[:,:,ch_num],curr_filter[:,:,ch_num])
            else:
                conv_map = self.conv_(x_train,curr_filter)
            feature_maps[:,:,filter_num] =conv_map
        return feature_maps
    def conv_(self,x_train,conv_filter):
        filter_size = conv_filter.shape[0]
        result = np.zeros((x_train.shape))
        for r in np.uint16(np.arange(filter_size/2,x_train.shape[0]-filter_size/2-2)):
            for c in np.uint16(np.arange(filter_size,x_train.shape[1]-filter_size/2-2)):
                curr_region = x_train[r:r+filter_size,c:c+filter_size]
                curr_result = curr_region * conv_filter
                conv_sum = np.sum(curr_result)
                result[r,c] = conv_sum
        final_result = result[np.uint16(filter_size/2):x_train.shape[0]-np.uint16(filter_size/2),np.uint16(filter_size/2):x_train.shape[1]-np.uint16(filter_size/2)]
        return final_result
    def relu(self,feature_map):
        arr =np.zeros(feature_map.shape)
        for i in range(feature_map.shape[-1]):
            for j in range(feature_map.shape[0]):
                for k in range(feature_map.shape[1]):
                    arr[j,k,i] = np.max(feature_map[j,k,i],0)
        return arr
    def pooling(self,feature_map,size=2,stride=2):
        arr = np.zeros((np.uint16((feature_map.shape[0] -size+1)/stride),np.uint16((feature_map.shape[1]-size+1)/stride),feature_map.shape[-1]))
        for dim in range(feature_map.shape[-1]):
            r =0
            for i in np.arange(0,feature_map.shape[0]-size-1,stride):
                c=0
                for j in np.arange(0,feature_map.shape[1]-size-1,stride):
                    arr[r,c,dim] = np.max(feature_map[i:i+size,j:j+size])
                    c = c +1
                r = r +1
        return arr
    def flatten(self,feature_map):
        dim =1
        for i in range(len(feature_map.shape)):
            dim *= feature_map.shape[i]
        return feature_map.reshape(dim,)
    def loss(self,y_hat,y):
        return -np.mean(np.log(y_hat[y]))
    def initialize(self,d0,d1,d2):
        x_train =self.x_train
        y_train =self.y_train
        w1 = 0.01*np.random.randn(d0,d1)
        b1 = np.zeros(d1)
        w2 = 0.01*np.random.randn(d1,d2)
        b2 = np.zeros(d2)
        print(x_train.shape)
        l1_feature_map_relu_pooling =self.pooling(self.relu(self.conv(x_train,self.l1_filter)))
        print(l1_feature_map_relu_pooling.shape)
        self.l2_filter = np.random.rand(3,5,5,l1_feature_map_relu_pooling.shape[-1])
        l2_feature_map_relu_pooling = self.pooling(self.relu(self.conv(l1_feature_map_relu_pooling,self.l2_filter)))
        print('before flatten',l2_feature_map_relu_pooling.shape)
        l2_feature_map_relu_pooling_flatten = self.flatten(l2_feature_map_relu_pooling)
        
        return l2_feature_map_relu_pooling_flatten,y_train,w1,b1,w2,b2
    def optimize(self,learningRate=0.005,steps=2000):
        x,y,w1,b1,w2,b2 = self.initialize(23217,128,1)
        
        costs =[]
        for i in range(steps):
            z1 = x.dot(w1)*b1
            a1 = np.maximum(z1,0)
            print(a1.shape)
            z2 = a1.dot(w2) +b2
            print(z2.shape)
            y_hat = self.sigmoid(z2)
            print('yhat',y_hat)
            if i%100 ==0:
                cost = self.loss(y_hat,y)
                print('cost after %d: %f' %(i,cost))
                costs.append(cost)
            
            y_hat[y] -=1
            e2 = y_hat/len(y_hat)
            dw2 = np.dot(a1,e2)
            db2 = np.sum(e2,axis=0)
            e1 = np.dot(e2,w2.T) * self.sigmoid_prime(z1)
            dw1 = np.dot(X.T,e1)
            db1 = np.sum(e1,axis=0)

            w1 -= learningRate*dw1
            b1 -= learningRate*db1
            w2 -= learningRate*dw2
            b2 -= learningRate*db2
        return w1,b1,w2,b2,costs

x_train = skimage.data.chelsea()
x_train = skimage.color.rgb2gray(x_train)
x_train = np.array(x_train)
print(x_train.shape)
l1_filter = np.zeros((2,3,3))
l1_filter[0,:,:] = np.array([[
    [-1,0,-1],
    [-1,0,-1],
    [-1,0,-1]
]])
l1_filter[1,:,:] = np.array([[
    [1,1,1],
    [0,0,0],
    [-1,-1,-1]
]])
y_train = np.array([0])
# l1_feature_map_relu_pooling = pooling(relu(conv(x_train,l1_filter)))
# l2_filter = np.random.rand(3,5,5,l1_feature_map_relu_pooling.shape[-1])
# l2_feature_map_relu_pooling = pooling(relu(conv(l1_feature_map_relu_pooling,l2_filter)))
cnn = ConvolutionalNeuralNetwork(x_train,y_train,l1_filter)
cnn.optimize()