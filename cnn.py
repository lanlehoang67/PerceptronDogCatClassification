import skimage.data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ConvolutionalNeuralNetwork():
    def __init__(self,x_train,l1_filter):
        self.x_train = x_train
        self.y_train = y_train
        self.l1_filter = l1_filter
        self.w = np.zeros((x_train.shape[0],1))
        self.b =0
    def sigmoid(self,input):
        return 1/(1+np.exp(-input))
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
    def optimize(self,learningRate=0.005,steps=2000):
        x_train =self.x_train
        y_train =self.y_train
        w = self.w
        b= self.b
        l1_feature_map_relu_pooling =self.pooling(self.relu(self.conv(x_train,self.l1_filter)))
        self.l2_filter = np.random.rand(3,5,5,l1_feature_map_relu_pooling.shape[-1])
        l2_feature_map_relu_pooling = self.pooling(self.relu(self.conv(l1_feature_map_relu_pooling,self.l2_filter)))
        l2_feature_map_relu_pooling_flatten = self.flatten(l2_feature_map_relu_pooling)
        A = self.relu(self.forward(l2_feature_map_relu_pooling_flatten,self.w,self.b))
        Y_hat = self.sigmoid(A,self.w,self.b)
        costs =[]
        for i in range(steps):
            dw,db,cost = self.backward(x_train,y_train,w,b)
            w = w - learningRate*dw
            b = b - learningRate*db
            if i%100 =0:
                print('cost after %i: %f',%i(i,cost))
                costs.append(cost)
        return w,b,cost

    def forward(self,input,w,b):
        return np.dot(input,w)+b
    def backward(self,X,Y,w,b):
        A = self.sigmoid(np.dot(w.T,X) +b)
        m = X.shape[1]
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A-Y)/m
        cost = (-1  / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return dw,db,cost
        
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
# l1_feature_map_relu_pooling = pooling(relu(conv(x_train,l1_filter)))
# l2_filter = np.random.rand(3,5,5,l1_feature_map_relu_pooling.shape[-1])
# l2_feature_map_relu_pooling = pooling(relu(conv(l1_feature_map_relu_pooling,l2_filter)))
cnn = ConvolutionalNeuralNetwork(x_train,l1_filter)
cnn.optimize()