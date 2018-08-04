import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
INPUT_PATH="/Activity-Recognition-from-Single-Chest-Mounted-Accelerometer/14.csv" #Took input from the 14th CSV File present. Please edit this accordingly, to feed the whole dataset or a particular CSV File
dataset=pd.read_csv(INPUT_PATH)
headers=["Serial Number", "X-axis", "Y-axis", "Z-axis", "Activity"]
def add_headers(dataset1, headers1):
    dataset1.columns=headers1
    return dataset1
dataset=add_headers(dataset, headers)
a=np.array(dataset)
target=a[:,4]
x=a[:,1]
y=a[:,2]
z=a[:,3]
am=np.sqrt(np.square(x)+np.square(y)+np.square(z))
window_size=52
stride=26
x_avg=[np.mean(x[i:i+window_size]) for i in range(0, len(x), stride) if i+window_size <= len(x)]
y_avg=[np.mean(y[i:i+window_size]) for i in range(0, len(y), stride) if i+window_size <= len(y)]
z_avg=[np.mean(z[i:i+window_size]) for i in range(0, len(z), stride) if i+window_size <= len(z)]
am_avg=[np.mean(am[i:i+window_size]) for i in range(0, len(am), stride) if i+window_size <= len(am)]
x_std=[np.std(x[i:i+window_size]) for i in range(0, len(x), stride) if i+window_size <= len(x)]
y_std=[np.std(y[i:i+window_size]) for i in range(0, len(y), stride) if i+window_size <= len(y)]
z_std=[np.std(z[i:i+window_size]) for i in range(0, len(z), stride) if i+window_size <= len(z)]
am_std=[np.std(am[i:i+window_size]) for i in range(0, len(am), stride) if i+window_size <= len(am)]
x_max=[np.max(x[i:i+window_size]) for i in range(0, len(x), stride) if i+window_size <= len(x)]
y_max=[np.max(y[i:i+window_size]) for i in range(0, len(y), stride) if i+window_size <= len(y)]
z_max=[np.max(z[i:i+window_size]) for i in range(0, len(z), stride) if i+window_size <= len(z)]
am_max=[np.max(am[i:i+window_size]) for i in range(0, len(am), stride) if i+window_size <= len(am)]
x_min=[np.min(x[i:i+window_size]) for i in range(0, len(x), stride) if i+window_size <= len(x)]
y_min=[np.min(y[i:i+window_size]) for i in range(0, len(y), stride) if i+window_size <= len(y)]
z_min=[np.min(z[i:i+window_size]) for i in range(0, len(z), stride) if i+window_size <= len(z)]
am_min=[np.min(am[i:i+window_size]) for i in range(0, len(am), stride) if i+window_size <= len(am)]
target1=[np.min(target[i:i+window_size]) for i in range(0, len(target), stride) if i+window_size <= len(target)]
dataset1=np.vstack((x_avg, y_avg, z_avg, am_avg, x_std, y_std, z_std, am_std, x_max, y_max, z_max, am_max, x_min, y_min, z_min, am_min, target1)).T
df=pd.DataFrame(dataset1)
df.to_csv("/home/sounak/ProcessedData.csv")
headers1=["Serial Number", "X Mean", "Y Mean", "Z Mean", "Am Mean", "X Std Deviation", "Y Std Deviation", "Z Std Deviation", "Am Std Deviation", "X Maximum", "Y Maximum", "Z Maximum", "Am Maximum", "X Minimum", "Y Minimum", "Z Minimum", "Am Minimum", "Activity"]
INPUT_PATH="/home/sounak/ProcessedData.csv"
dataset=pd.read_csv(INPUT_PATH)
dataset=add_headers(dataset, headers1)
def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y=train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
    return train_x, test_x, train_y, test_y
train_x, test_x, train_y, test_y = split_dataset(dataset, 0.9, headers1[1:-1], headers1[-1])
print ("Train_x shape :: ", train_x.shape)
print ("Test_x shape :: ", test_x.shape)
print ("Train_y shape :: ", train_y.shape)
print ("Test_y shape :: ", test_y.shape)
def random_forest_classifier(features, target):
    clf=RandomForestClassifier()
    clf.fit(features, target)
    return clf
trained_model=random_forest_classifier(train_x, train_y)
print ("Trained model :: ", trained_model)
predictions=trained_model.predict(test_x)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
    print ("Actual Outcome :: {} and Predicted Outcome :: {} ".format(list(test_y)[i], predictions[i]))
print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
print ("Test Accuracy :: ", accuracy_score(test_y, predictions))
