from sklearn.datasets import load_iris 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np

(data, target) = load_iris(return_X_y = True)
model = KNeighborsClassifier()

accuracy = 0

for i in range(len(target)):
    data_train, target_train = [np.delete(d, i, axis = 0) for d in [data, target]]
    model.fit(data_train, target_train)
    prediction = model.predict(data[i].reshape(1,-1))
    if prediction == target[i]:
        accuracy += 1 / len(target)

print(accuracy)
