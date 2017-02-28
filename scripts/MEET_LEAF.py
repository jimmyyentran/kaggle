# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 02:23:21 2017

@author: jkim
"""

## Shapelets for time series classification 

import itertools
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib as mpl

import matplotlib.pyplot as plt

from skimage import measure
import scipy.ndimage as ndi

# from pylab import rcParams
mpl.rcParams['figure.figsize'] = (6, 6) 

## Step 1: Converting an object contour to a 1D signal 
def draw_leaf(image):
    img = mpimg.imread(image)
    cy, cx = ndi.center_of_mass(img)
    return img, (cx, cy)
    
def get_contour(img, thresh=.8):
    contours = measure.find_contours(img, thresh)
    return max(contours, key=len)  # Take longest one
    
def convert_to_1d(file, sample=250, thresh=.8, plot=False, norm=True):
    img, (cx, cy) = draw_leaf(file)
    contour = get_contour(img, thresh)
    distances = [manhattan_distance([cx, cy], [contour[i][0], contour[i][1]]) for i in range(0, len(contour), sample)]
    distances.extend(distances)
    if plot:
        f, axarr = plt.subplots(2, sharex=False) 
        axarr[0].imshow(img, cmap='Set3')
        axarr[0].plot(contour[::,1], contour[::,0], linewidth=0.5)
        axarr[0].scatter(cx, cy)
        axarr[1].plot(distances)
        plt.show()
    if norm: 
        return np.divide(distances, max(distances))
    else:
        return distances  # Extend it twice so that it is cyclic

def manhattan_distance(a, b, min_dist=float('inf')):
    dist = 0
    for x, y in zip(a, b):
        dist += np.abs(float(x)-float(y))
        if dist >= min_dist: return None
    return dist

data = []
for i in range(1, 1584):
 #  distance[i] = convert_to_1d('/Users/jkim/Documents/leaf_image/images/i.jpg', plot=True, norm=1)
    data.append(convert_to_1d('/Users/jkim/Documents/leaf_image/images/'+str(i)+'.jpg', plot = True, norm=1))
#distances1 = convert_to_1d('/Users/jkim/Documents/leaf_image/images/1.jpg', plot=True, norm=1)
#distances2 = convert_to_1d('/Users/jkim/Documents/leaf_image/images/2.jpg', plot=True, norm=1)
#distances3 = convert_to_1d('/Users/jkim/Documents/leaf_image/images/3.jpg', plot=True, norm=1)
#distances4 = convert_to_1d('/Users/jkim/Documents/leaf_image/images/4.jpg', plot=True, norm=1)
#distances5 = convert_to_1d('/Users/jkim/Documents/leaf_image/images/5.jpg', plot=True, norm=1)

#data.append((convert_to_1d('/Users/jkim/Documents/leaf_image/images/'+str(number)+'.jpg', plot=0), leaf_map[name]))

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

SIM = []
#SIM[0] = similar(data[1], data[1])
for i in range(1, 1584):
    SIM.append(similar(data[1], data[i]))

#similar(distances1, distances1)
#similar(distances1, distances2)
#similar(distances1, distances3)
#similar(distances1, distances4)
#similar(distances1, distances5)

##########################  csv file loading. #########################
import pandas as pd
import numpy as np

train_data = pd.read_csv("/Users/jkim/Documents/leaf_image/train.csv") #, nrows=1)
test_data = pd.read_csv("/Users/jkim/Documents/leaf_image/test.csv") #, nrows=1)

#np.savetxt('train_data.csv', ())
#import csv
#myfile_TRAIN_DATA = open("/Users/jkim/Documents/leaf_image/TRAIN", 'wb') #, nrows=1)
#TRAIN_DATA = csv.writer(myfile_TRAIN_DATA, quoting=csv.QUOTE_ALL)

DATA = pd.concat([train_data, test_data])
DATA = DATA.sort(['id'], ascending=[True])


#DATA = pd.merge(train_data, test_data, on=['id'])


len(train_data)
train_data.shape
train_data.index
train_data.columns


#for val in range(1,len(SIM)):
#    train_data.ix[val] = SIM[val]

#for val in range(1, len(SIM)):
#    DATA.ix[val] = SIM[val]

#SIM = pd.DataFrame(SIM)

F_DATA = DATA
F_DATA = F_DATA.sort(['id'], ascending=[True])

F_DATA =DATA.join(SIM)
#remain = [p for p in F_DATA.id if p in train_data.id]

#train = []
#for p in remain:
#    train.append(F_DATA[p])
#train = pd.DataFrame(train)

train = F_DATA[F_DATA.id.isin(train_data.id)]
test = F_DATA[F_DATA.id.isin(test_data.id)]

#y_train_orig = train.species
y_train_orig = train['species']

my_cols = set(train.columns)
my_cols.remove('species')
my_cols = list(my_cols)

X_train_orig = train[my_cols]


y_test_orig = test['species']
y_train_orig = train['species']

len(numpy.unique(list(y_train_orig)))

my_cols = set(test.columns)
my_cols.remove('species')
my_cols = list(my_cols)

X_test_orig = test[my_cols]

test_col = X_test_orig.columns
train_col = X_train_orig.columns

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB

sc = StandardScaler()


# Importance rate calculation using RandomForestClassifier. 
from sklearn.ensemble import RandomForestClassifier 
feat_labels = X_train_orig.columns 

forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

forest.fit(X_train_orig, y_train_orig)


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print(importances)
print(indices)




#for f in range(X_train_orig.shape[1]):
#    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
#X_train_orig = forest.transform(X_train_orig, threshold=.05)
#X_test_orig = forest.transform(X_test_orig, threshold=.05)
#X_train_orig 



# put into the data frame. for X-test data 
X_test_orig = pd.DataFrame(sc.fit_transform(X_test_orig))
# interpolate the missing values using two values sides. 
X_train_orig = X_train_orig.interpolate()
# put into the data frame for X-train data. 
X_train_orig = pd.DataFrame(sc.fit_transform(X_train_orig))
# test column names
X_test_orig.columns = test_col
# train column names 
X_train_orig.columns = train_col

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_orig, y_train_orig, test_size=0.25, random_state=0)



## learning curves of KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier

print("KNeighborsClassifier")
train_sizes, train_scores, test_scores = learning_curve(estimator=KNeighborsClassifier(n_neighbors=3), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)
Y = knn.predict(X_train)
Y_pred = knn.predict(X_test)
#print(knn.score(X_train, Y))
print(knn.score(X_train, y_train))







#coeff_df = DataFrame(train_df.columns.delete(0))
#coeff_df.columns = ['Features']
#coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
#print(coeff_df)



#from sklearn.metrics import classification_report
#from sklearn import metrics
#y_true, y_pred = y_test, clf.predict(X_test)
#print(classification_report(y_true, y_pred))
#y_pred = clf.predict(X_test).astype(int)
