import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
from sklearn.utils import shuffle
from scipy.special import expit

def sigmoid(x):
    return expit(x) #exponantial function
def sigmoid_der(x):
    return sigmoid(x) * (1- (sigmoid(x)))
def train():
    img_dir = "D:\Program Files (x86)\Microsoft Visual Studio\Projeler\Spyder-Python\Deep.L.Hw3\\train\cannon" 
    data_path = os.path.join(img_dir,'*jpg') 
    cannon_file = glob.glob(data_path) 
    cannon = [] 
    for f1 in cannon_file: 
        img = cv2.imread(f1) 
        img = cv2.resize(img, (64,64))
        cannon.append(img) 
    
    img_dir = "D:\Program Files (x86)\Microsoft Visual Studio\Projeler\Spyder-Python\Deep.L.Hw3\\train\cellphone" 
    data_path = os.path.join(img_dir,'*jpg') 
    cellphone_file = glob.glob(data_path) 
    cellphone = [] 
    for f1 in cellphone_file: 
        img = cv2.imread(f1) 
        img = cv2.resize(img, (64,64))
        cellphone.append(img) 
    
    t = []
    for images in cannon:
        t.append((images.flatten() , 0))
        
    for images in cellphone:
        t.append((images.flatten() , 1))

    y_train = []
    y_label = []
    for index in t:
        y_train.append(index[0])
        y_label.append(index[1])
    
    for i in range(100):
        y_train[i] = np.append(y_train[i], 1)
        
    y_train, y_label = shuffle(y_train, y_label)
    
    weight = np.random.random((len(y_train[0])))
    
    def trainPerceptron(inputs, t, weights, rho, iterNo):
        for epoch in range(iterNo):
            inputs = y_train
            # feedforward step 1
            XW = np.dot(y_train,weights)
            XW = np.float64(XW)
            # feedforward step 2
            z = sigmoid(XW)
            # backpropagation step 
            
            # error_out = ((1 / 2) * (np.power((z - t), 2)))
            error = z - t
        
            # backpropagation step 2
            dcost_dpred = error
            dpred_dz = sigmoid_der(z) 
        
            z_delta = dcost_dpred * dpred_dz
        
            inputs = np.transpose(y_train)
            weights -= rho * np.dot(inputs, z_delta)
        return weights
    
    deneme = []
    weight = trainPerceptron(deneme, y_label, weight, 0.001, 1000)
    np.save('weights.npy', weight) # save

def test():
    img_dir = "D:\Program Files (x86)\Microsoft Visual Studio\Projeler\Spyder-Python\Deep.L.Hw3\\test\cannon" 
    data_path = os.path.join(img_dir,'*jpg') 
    test_cannon_file = glob.glob(data_path) 
    for f1 in test_cannon_file: 
        img = cv2.imread(f1) 
        img = cv2.resize(img, (64,64))
        test_cannon = img
    
    img_dir = "D:\Program Files (x86)\Microsoft Visual Studio\Projeler\Spyder-Python\Deep.L.Hw3\\test\cellphone" 
    data_path = os.path.join(img_dir,'*jpg') 
    test_cellphone_file = glob.glob(data_path) 
    for f1 in test_cellphone_file: 
        img = cv2.imread(f1) 
        img = cv2.resize(img, (64,64))
        test_cellphone = img
    
    plt.imshow(test_cannon)
    test_cannon = test_cannon.flatten()
    test_cellphone = test_cellphone.flatten()
    
    test_cannon = np.append(test_cannon, 1)
    test_cellphone = np.append(test_cellphone, 1)
    
    weights = np.load('weights.npy') # load
    
    def testPerceptron(sample_test, weights):
        summation = np.dot(sample_test, weights)
        summation = sigmoid(summation)
        if summation > 0.5:
            return("Test image -> Cellphone", summation)
        else:
            return("Test image -> Cannon", summation)
    
    
    print(testPerceptron(test_cannon, weights))
    
# train()  
test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
