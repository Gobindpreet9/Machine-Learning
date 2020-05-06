# Goal: To implement a Decision Tree similar to CART algorithm
import numpy as np


def gini_impurity(labels):
    if not labels:  # Check if labels is empty
        return 0
    else:
        number_of_labels = np.unique(labels, return_counts=True)[
            1]  # 2nd array returned gives the count itself, 1st array the labels
        fractions = number_of_labels / float(len(labels))
        return 1 - np.sum(fractions ** 2)


def entropy(labels):
    if not labels:
        return 0
    else:
        number_of_labels = np.unique(labels, return_counts=True)
        fractions = number_of_labels / float(len(labels))
        return - np.sum(fractions * np.log2(fractions))


criterion_function = {
    'gini': gini_impurity,
    'entropy': entropy
}


def weighted_impurity(groups, critertion='gini'):
    """
    Calculate weighted impurity of children after a split
    :param groups: list of children
    :param critertion: gini for Gini Impurity or entropy for
    :return: float weighted impurity for the split
    """
    total = sum(len(groups))
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[critertion](group)
    return weighted_sum


def split_node(X, y, index, value):
    """
    Split dataset on the basis of a feature and value
    :param X: numpy.ndarray, dataset feature
    :param y: numpy.ndarray, dataset target
    :param index: int, index of the feature used for splitting
    :param value: value of feature used for splitting
    :return: list, list, left and right child in the format of [X, y]
    """
    x_index = X[:, index]
    # if the feature is numerical
    if X[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    else:
        # feature is categorical
        mask = x_index == value
    left = [X[~mask, :], y[~mask]]  # :/
    right = [X[mask, :], y[~mask]]  # :/
    return left, right


def get_best_split(X, y, criterion):
    """
    Obtain best splitting point
    :param X: numpy.ndarray, dataset feature
    :param y: numpy.ndarray, dataset target
    :param criterion: gini or entropy
    :return: dict {indx: , value:0, children: left and right }
    """
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(X[0])):  # iterate over number of columns
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_value, best_score, best_index, children = value, impurity, index, groups
    return {'index': best_index, 'value': best_value, 'children': children}


def get_leaf(labels):
    # Major class is returned as the leaf
    return np.bincount(labels).argmax()


def split(node, max_depth, min_size, depth, criterion):
    """
    Split children of a node to construct new nodes or assign terminal nodes
    :param node: dictionary of children
    :param max_depth:
    :param min_size:
    :param depth: current depth
    :param criterion: 'gini' or 'entropy'
    """
    left, right = node['children']
    del (node['children'])
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])  # why not get_leaf(right)?
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['light'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])  # why not get_leaf(right)?
    else:
        result = get_best_split(left[0], left[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['left'] = result
        split(node['left'], max_depth, min_size, depth + 1, criterion)
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])  # why not get_leaf(right)?
    else:
        result = get_best_split(right[0], right[1], criterion)
        result_left, result_right = result['children']
        if result_left[1].size == 0:
            node['left'] = get_leaf(result_right[1])
        elif result_right[1].size == 0:
            node['right'] = get_leaf(result_left[1])
        else:
            node['right'] = result
        split(node['right'], max_depth, min_size, depth + 1, criterion)


def train_tree(X_train, y_train, max_depth, min_size, criterion='gini'):
    """
    Construction of tree starts here
    :param X_train: list
    :param y_train: list
    :param max_depth:
    :param min_size:
    :param criterion: 
    """
    x = np.array(X_train)
    y = np.array(y_train)
    root = get_best_split(x, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root


# X_train = [
#     [6, 7], [2, 4], [7, 2], [3, 6], [4, 7], [5, 2], [1, 6], [2, 0], [6, 3], [4, 1]
# ]
# y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
#
# X_train = [
#     [6, 7], [2, 4], [7, 2], [3, 6], [4, 7], [5, 2], [1, 6], [2, 0], [6, 3], [4, 1]
# ]
# y_train = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#
# tree_sk = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
# tree_sk.fit(X_train, y_train)
#
# # Visualizing Tree
# export_graphviz(tree_sk, out_file='tree.dot', feature_names=['X1', 'X2'], impurity=False, filled=True,
#                 class_names=['0', '1'])

