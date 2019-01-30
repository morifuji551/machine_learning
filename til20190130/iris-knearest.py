from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

max_k = 15

class Data():
    def __init__(self, data, label):
        self.data = data 
        self.label = label

class K_NEAREST():
    def __init__(self, Data_array, k): #Data_array is an array of Data objects.
        self.Data_array = Data_array
        self.distance = self.calc_distance()
        self.k = k
        self.prediction = self.predict()

    def calc_distance(self): 
        distance = np.zeros((len(self.Data_array), len(self.Data_array)))
        for i in range(len(self.Data_array)):
            for j in range(len(self.Data_array)):
                err = self.Data_array[i].data - self.Data_array[j].data
                distance[i][j] = np.linalg.norm(err)
        return distance
    
    def predict(self):
        top_k_index = np.zeros((len(self.Data_array),self.k))
        predictions = np.zeros(len(self.Data_array))
        for i in range(len(self.Data_array)):
            for j in range(self.k):
                index = np.where(self.distance[i] == np.sort(self.distance[i])[j+1])[0][0] #一番目は自分自身の距離0であるため、j+1とする
                top_k_index[i][j] = self.Data_array[index].label #j番目に距離が小さいもののラベルを取得する
            number_of_zeros = np.sum(top_k_index[i] == 0)
            number_of_ones = np.sum(top_k_index[i] == 1)
            number_of_twos = np.sum(top_k_index[i] == 2)
            predictions[i] = np.argmax((number_of_zeros, number_of_ones, number_of_twos))
        return predictions
    
    def accuracy(self):
        check_answer = np.zeros(len(self.Data_array))
        for i in range(len(self.Data_array)):
            if self.prediction[i] == self.Data_array[i].label:
                check_answer[i] = 1
        accuracy = np.sum(check_answer) / len(self.Data_array)
        return accuracy


    
(data, target) = load_iris(return_X_y = True)
data, target = [np.array(d) for d in [data, target]]
Data_array = [Data(data[i], target[i]) for i in range(len(target))]

result_list = np.zeros((max_k, 2))

for k in range(max_k):
    k_nearest = K_NEAREST(Data_array, k)
    result_list[k][0] = k
    result_list[k][1] = k_nearest.accuracy()

plt.plot(result_list.T[0], result_list.T[1])
plt.show()