import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns


def percentage(decimal):
    return decimal * 100


def get_grade(score):
    # https://en.wikipedia.org/wiki/Academic_grading_in_Portugal
    if 0 <= score < 3:
        return 0
        # Poor result
    elif 3 <= score < 9:
        return 1
        # Weak result
    elif 9 <= score < 13:
        return 2
        # Sufficient result
    elif 13 <= score < 15:
        return 3
        # Good result
    elif 15 <= score < 17:
        return 4
        # Very good result
    else:
        return 5
        # Excellent result


def get_feature_columns(data):
    cols = data.columns.tolist()
    cols.remove('grade')
    return cols


def get_x(data, columns):
    return data[columns].values


def get_y(data):
    return data['grade'].values


def get_tsne_data(data, _x):
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(_x)
    data_tsne = data.copy(deep=True)
    data_tsne['x-tsne'] = tsne_results[:, 0]
    data_tsne['y-tsne'] = tsne_results[:, 1]
    return data_tsne


def get_clusters(_x):
    km = KMeans(n_clusters=5)
    km.fit(_x)
    return km.predict(_x)


student_data = pd.read_csv(open('data/student/student-mat.csv', 'r'), sep=';')

# PRE-PROCESSING

sns.set()
np.random.seed(42)
student_data = pd.get_dummies(student_data)

student_data['grade'] = student_data['G3'].apply(lambda score: get_grade(score))


# CLUSTERING

feature_columns = get_feature_columns(student_data)
x = get_x(student_data, feature_columns)

clusters = get_clusters(x)
student_data['cluster'] = clusters

feature_columns = get_feature_columns(student_data)
x = get_x(student_data, feature_columns)
y = get_y(student_data)

# PLOTTING

student_tsne = get_tsne_data(student_data, x)
plt.scatter(student_tsne['x-tsne'], student_tsne['y-tsne'], c=clusters, cmap='Set3')

# REGRESSION PARAM TUNING

plt.figure(figsize=(16, 10))
c_set = [1, 100, 1000]
penalty_set = ['l1', 'l2']

ctr = 1


for ix, c_param in enumerate(c_set):
    print('C Loop: {}'.format(ix))
    for jx, penalty_param in enumerate(penalty_set):
        print('Penalty Loop: {}'.format(jx))

        r_accuracies_train = []
        r_accuracies_test = []

        # Cross validation in a loop.
        for i in range(25):
            print('Validation Loop: {}'.format(i))
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)

            regression_model = linear_model.LogisticRegression(penalty=penalty_param, C=c_param)

            regression_model.fit(train_x, train_y)

            r_accuracies_train.append(regression_model.score(train_x, train_y))
            r_accuracies_test.append(regression_model.score(test_x, test_y))

        plt.subplot(len(c_set), len(penalty_set), ctr)
        ctr += 1

        assert len(r_accuracies_train) == len(r_accuracies_test)

        plt.plot(range(len(r_accuracies_train)), r_accuracies_train, label='Train')
        plt.plot(range(len(r_accuracies_test)), r_accuracies_test, label='Test')

        plt.ylim(0.8, 1.0)

        plt.legend(loc='upper left')
        plt.title("C({}), Penalty({})".format(c_param, penalty_param))


# CROSS VALIDATION

accuracies = {
    'tree': {
        'test': [],
        'train': []
    },
    'regression': {
        'test': [],
        'train': []
    }
}

f, axarr = plt.subplots(2, 2, sharex='all')

axarr[0, 0].set_title('Tree Test')
axarr[0, 0].axis([1, 10, 40, 100])
axarr[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

axarr[0, 1].set_title('Tree Train')
axarr[0, 1].axis([1, 10, 40, 100])
axarr[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

axarr[1, 0].set_title('Regression Test')
axarr[1, 0].axis([1, 10, 40, 100])
axarr[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

axarr[1, 1].set_title('Regression Train')
axarr[1, 1].axis([1, 10, 40, 100])
axarr[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

for ax in axarr.flat:
    ax.set(xlabel='Pass number', ylabel='Accuracy (%)')
for ax in axarr.flat:
    ax.label_outer()

tree_test_acc = []
tree_train_acc = []
regression_test_acc = []
regression_train_acc = []

for i in range(11):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=np.random.randint(100))

    regression_model = linear_model.LogisticRegression(penalty='l1', C=100)
    tree_model = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)

    tree_model.fit(train_x, train_y)
    regression_model.fit(train_x, train_y)

    tree_test_acc.append(percentage(tree_model.score(test_x, test_y)))
    tree_train_acc.append(percentage(tree_model.score(train_x, train_y)))
    regression_test_acc.append(percentage(regression_model.score(test_x, test_y)))
    regression_train_acc.append(percentage(regression_model.score(train_x, train_y)))


axarr[0, 0].plot(
    tree_test_acc,
    color='green'
)
axarr[0, 1].plot(
    tree_train_acc,
    color='blue'
)
axarr[1, 0].plot(
    regression_test_acc,
    color='orange'
)
axarr[1, 1].plot(
    regression_train_acc,
    color='red'
)

# tree_predictions = tree_model.predict(test_x)
#
# print('TREE')
# print('Accuracy:\n{}'.format(accuracy_score(test_y, tree_predictions)))
# print('Con Matrix:\n{}'.format(confusion_matrix(test_y, tree_predictions)))
# print('Class Report:\n{}'.format(classification_report(test_y, tree_predictions)))
#
# regression_predictions = regression_model.predict(test_x)
#
# print('REGRESSION')
# print('Accuracy:\n{}'.format(accuracy_score(test_y, regression_predictions)))
# print('Con Matrix:\n{}'.format(confusion_matrix(test_y, regression_predictions)))
# print('Class Report:\n{}'.format(classification_report(test_y, regression_predictions)))
#
# t_scores = cross_val_score(tree_model, test_x, test_y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (t_scores.mean(), t_scores.std() * 2))
# r_scores = cross_val_score(regression_model, test_x, test_y, cv=5)
# print("Accuracy: %0.2f (+/- %0.2f)" % (r_scores.mean(), r_scores.std() * 2))
#
# tree_accuracy = {
#     'test': [],
#     'train': []
# }
# regression_accuracy = {
#     'test': [],
#     'train': []
# }
#
# for i in range(10):
#     print('PASS {}'.format(i + 1))
#

#

#
#     tree_model.fit(train[0], train[1])
#     regression_model.fit(train[0], train[1])
#
#     # print_score(regression_model, train, test)
#     # print_score(tree_model, train, test)
#
#     tree_accuracy['train'].append(get_model_score(tree_model, train))
#     tree_accuracy['test'].append(get_model_score(tree_model, test))
#
#     regression_accuracy['train'].append(get_model_score(regression_model, train))
#     regression_accuracy['test'].append(get_model_score(regression_model, test))


# tree.export_graphviz(tree_model, out_file="graphs/tree.dot")
