import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if not os.path.exists('./Skin_NonSkin.txt'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    urllib.request.urlretrieve(url, './Skin_NonSkin.txt')

df = pd.read_csv('Skin_NonSkin.txt', sep='\t', names=['B', 'G', 'R', 'skin'])
df.head()

df.isna().sum()

feature = df[df.columns[~df.columns.isin(['skin'])]]  # Except Label
label = (df[['skin']] == 1) * 1  # Converting to 0 and 1 (this col has values 1 and 2)
feature = feature / 255.  # Pixel values range from 0-255 converting between 0-1

feature.head()
label.head()

alldf = pd.concat([feature, label], sort=True, axis=1)
alldf

sample = alldf.sample(1000)

onlybgr = sample[sample.columns[~sample.columns.isin(['skin'])]]

sns.pairplot(onlybgr)

sample_ = sample.copy()
sample_['skin'] = sample.skin.apply(lambda x: {1: 'skin', 0: 'not skin'}.get(x))
sns.pairplot(sample_, hue="skin")

sns.pairplot(onlybgr, kind="reg")

# Lets see how many 0s and 1s
(label == 0).skin.sum(), (label == 1).skin.sum()

x = feature.values
y = label.values

# We will keep fix test and take 5 cross validation set
# so we will have five different data set
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)

# Lets see the size of xtrain, xtest
len(xtrain), len(xtest)

# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks

data = np.concatenate([xtrain, ytrain], axis=1)
# Observe the shape of array
xtrain.shape, ytrain.shape, data.shape

# Divide our data to 5 chunks
chunks = np.split(data, 5)

datadict = {'fold1': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold2': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold3': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold4': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None}, 'test': {'x': xtest, 'y': ytest}},
            'fold5': {'train': {'x': None, 'y': None}, 'val': {'x': None, 'y': None},
                      'test': {'x': xtest, 'y': ytest}}, }

for i in range(5):
    datadict['fold' + str(i + 1)]['val']['x'] = chunks[i][:, 0:3]
    datadict['fold' + str(i + 1)]['val']['y'] = chunks[i][:, 3:4]

    idx = list(set(range(5)) - set([i]))
    X = np.concatenate(itemgetter(*idx)(chunks), 0)
    datadict['fold' + str(i + 1)]['train']['x'] = X[:, 0:3]
    datadict['fold' + str(i + 1)]['train']['y'] = X[:, 3:4]


def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def readpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


writepickle(datadict, 'data.pkl')

data = readpickle('data.pkl')


fold1 = data['fold1']
fold1_train = fold1['train']
fold1_val = fold1['val']
fold1_test = fold1['test']

fold2 = data['fold2']
fold2_train = fold2['train']
fold2_val = fold2['val']
fold2_test = fold2['test']

fold3 = data['fold3']
fold3_train = fold3['train']
fold3_val = fold3['val']
fold3_test = fold3['test']

fold4 = data['fold4']
fold4_train = fold4['train']
fold4_val = fold4['val']
fold4_test = fold4['test']

fold5 = data['fold5']
fold5_train = fold5['train']
fold5_val = fold5['val']
fold5_test = fold5['test']

##### Logistic Regression #####

# Performs logistic regression using the sklearn library and appends the accuracy score for test
# and validation sets to an array
def logistic_regression(kfold, xtrain, ytrain, xtest, ytest, xval, yval, score_test, score_val):
    classifier = LogisticRegression()
    classifier.fit(xtrain, ytrain.ravel())

    print('Accuracy of logistic regression classifier fold: {} on test set: {:.3f}'.format(kfold+1, classifier.score(xtest, ytest)))
    print('Accuracy of logistic regression classifier fold: {} on validation set: {:.3f}'.format(kfold+1, classifier.score(xval, yval)))

    score_test.append('{:.3f}'.format(classifier.score(xtest, ytest)))
    score_val.append('{:.3f}'.format(classifier.score(xval, yval)))


# Given an array of values and number of values, calculates and returns the average of the values
def find_average(values, num_values):
    sum = 0
    for i in values:
        sum = sum + float(i)

    return sum/num_values

# Prints the accuracy table given an array of accuracy scores on the test sets and accuracy scores on the
# validation sets
def print_table(score_test, score_val):
    print('-----------------------')
    print('     |  ACCURACY')
    print('FOLD | VAL   | TEST')
    print('-----------------------')
    print('     |       |')
    for i in range(5):
        print(str(i+1) +'    | ' + score_test[i] + ' | ' + score_val[i])
    print('-----------------------')
    print("AVG | " + '{:.3f}'.format(find_average(score_test, 5)) + '  | ' + '{:.3f}'.format(find_average(score_val, 5)))

# Plot accuracy data
def plot(score_test, score_val):
     # data to plot
     n_groups = 5

     # create plot
     fig, ax = plt.subplots()
     index = np.arange(n_groups)
     bar_width = 0.35
     opacity = 0.8

     rects1 = plt.bar(index, score_test, bar_width,
                      alpha=opacity,
                      color='b',
                      label='Test Set')

     rects2 = plt.bar(index + bar_width, score_val, bar_width,
                      alpha=opacity,
                      color='g',
                      label='Validation Set')

     plt.xlabel('K-Folds')
     plt.ylabel('Accuracy Score')
     plt.title('K-Folds Accuracy Scores')
     plt.xticks(index + bar_width, ('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'))
     plt.legend(loc='best')

     plt.tight_layout()
     plt.show()



# Assign x train, y train, x test, y test, x val, y val based on which k-fold we are on
score_test = []
score_val = []
for i in range(5):
    if i is 0:
        xtrain, ytrain = fold2_train['x'] + fold3_train['x'] + fold4_train['x'] + fold5_train['x'], fold2_train['y'] + \
                         fold3_train['y'] + fold4_train['y'] + fold5_train['y']
        xval, yval = fold1_val['x'], fold1_val['y']
        xtest, ytest = fold1_test['x'], fold1_test['y']

    if i is 1:
        xtrain, ytrain = fold1_train['x'] + fold3_train['x'] + fold4_train['x'] + fold5_train['x'], fold1_train['y'] + \
                         fold3_train['y'] + fold4_train['y'] + fold5_train['y']
        xval, yval = fold2_val['x'], fold2_val['y']
        xtest, ytest = fold2_test['x'], fold2_test['y']

    if i is 2:
        xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold4_train['x'] + fold5_train['x'], fold1_train['y'] + \
                         fold2_train['y'] + fold4_train['y'] + fold5_train['y']
        xval, yval = fold3_val['x'], fold3_val['y']
        xtest, ytest = fold3_test['x'], fold3_test['y']

    if i is 3:
        xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold3_train['x'] + fold5_train['x'], fold1_train['y'] + \
                         fold2_train['y'] + fold3_train['y'] + fold5_train['y']
        xval, yval = fold4_val['x'], fold4_val['y']
        xtest, ytest = fold4_test['x'], fold4_test['y']

    if i is 4:
        xtrain, ytrain = fold1_train['x'] + fold2_train['x'] + fold3_train['x'] + fold4_train['x'], fold1_train['y'] + \
                         fold2_train['y'] + fold3_train['y'] + fold4_train['y']
        xval, yval = fold5_val['x'], fold5_val['y']
        xtest, ytest = fold5_test['x'], fold5_test['y']

    # Preform logistic regression and get accuracy for each k-fold
    logistic_regression(i, xtrain, ytrain, xtest, ytest, xval, yval, score_test, score_val)


# Print accuracy table
print_table(score_test, score_val)

# Plot accuracy visualization
plot(list(map(float, score_test)), list(map(float, score_val)))
