from sklearn.datasets import load_iris
import numpy as np 

class Data():
    def __init__(self, data, target, label_tmp):
        self.data = data 
        self.target = target #answer label
        self.label = label_tmp #temporary label
    
    def set_label(self, new_label):
        self.label = new_label

class Compare_Center():
    def __init__(self):
        self.new_center = 1 #temporary value
        self.old_center = 0 #temporary value 
    
    def set_new_center(self, center):
        self.old_center = self.new_center
        self.new_center = center
    
    def Compare(self):
        return np.all(self.new_center == self.old_center)

class KMEANS():
    def __init__(self, Data_array):
        self.Data_array = Data_array #Data_array is an array of Data objects.
        self.center = self.calc_center()
    
    def calc_center(self): #center is an array of size(3,1)
        num_data = np.zeros(3)
        center = np.zeros((3, len(self.Data_array[0].data))) # center of data
        for label in range(3):
            for Data in self.Data_array:
                if Data.label == label:
                    num_data[label] += 1
                    center[label] += Data.data
            center[label] /= num_data[label]
        return center 

    def alloc_new_label(self):
        distance = np.zeros(3)
        for Data in self.Data_array:
            for label in range(3):
                err = Data.data - self.center[label]
                distance[label] = np.linalg.norm(err)
            new_label = np.argmin(distance)
            Data.set_label(new_label)
        
(iris_data, target) = load_iris(return_X_y = True)
num_data = len(target)
target_init = np.array([i % 3 for i in range(num_data)]) #designate initial value 

Data_array = [Data(iris_data[i], target[i], target_init[i]) for i in range(num_data)]
compare_center = Compare_Center()

while not(compare_center.Compare()):
    kmeans = KMEANS(Data_array)
    kmeans.alloc_new_label()
    compare_center.set_new_center(kmeans.center)

for data in Data_array:
    print(data.label)





