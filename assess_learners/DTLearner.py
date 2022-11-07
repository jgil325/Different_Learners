import pandas as pd
import numpy as np


class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        """
        :return: The UGA username of the student
        :rtype: str
        """
        return "jg93593"  # replace ingrid with your UGA MyID username

    def build_tree(self, dataX, dataY):
        if dataX.shape[0] <= self.leaf_size or len(np.unique(dataX[:, -1])) == 1:
            return np.array([-1, dataY.mean(), np.nan, np.nan])
        else:
            num_X = dataX.shape[1]
            corr_of_columns = [abs(np.corrcoef(dataX[:, i], dataY)[0, 1]) for i in range(num_X)]
            best_column = np.nanargmax(corr_of_columns)
            SplitVal = np.median(dataX[:, best_column])
            if (SplitVal == np.amax(dataX[:, best_column])) or (SplitVal == np.amin(dataX[:, best_column])):
                return np.array([-1, dataY.mean(), np.nan, np.nan])
            split_condition1 = dataX[:, best_column] <= SplitVal
            split_condition2 = dataX[:, best_column] > SplitVal
            left_tree = self.build_tree(dataX[split_condition1], dataY[split_condition1])
            right_tree = self.build_tree(dataX[split_condition2], dataY[split_condition2])
            left_tree_size = left_tree.ndim
            if left_tree_size > 1:
                self.root = np.array([best_column, SplitVal, 1, left_tree.shape[0] + 1])
            elif left_tree_size == 1:
                self.root = np.array([best_column, SplitVal, 1, 2])
            return np.vstack((self.root, left_tree, right_tree))

    def add_evidence(self, dataX, dataY):
        self.tree = self.build_tree(dataX, dataY)
        if self.verbose:
            print(self.tree)

    def query(self, dataX_test):
        trainY = []
        for row in dataX_test:
            i = 0
            while i < self.tree.shape[0]:
                feature_ind = int(self.tree[i, 0])
                if feature_ind == -1:
                    break
                if row[feature_ind] > self.tree[i, 1]:
                    i += int(self.tree[i, 3])
                elif row[feature_ind] <= self.tree[i, 1]:
                    i += 1
            if feature_ind >= 0:
                trainY.append(np.nan)
            else:
                trainY.append(self.tree[i, 1])
        return trainY
