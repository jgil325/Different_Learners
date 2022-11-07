import DTLearner as dt
import LinRegLearner as lrl
import numpy as np
import random


class BagLearner(object):
    def __init__(self, learner=dt.DTLearner, kwargs=None, bags=20, boost=False, verbose=False):
        if kwargs is None:
            kwargs = {"argument1": 1, "argument2": 2}
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(self.learner(**self.kwargs))

    def author(self):
        """
        :return: The UGA username of the student
        :rtype: str
        """
        return "jg93593"  # replace ingrid with your UGA MyID username

    def add_evidence(self, dataX, dataY):
        self.dataX = dataX
        self.dataY = dataY

    def query(self, test_dataX):
        all_Y = []
        for lr in self.learners:
            data_size = self.dataX.shape[0]
            rand_ind = np.random.choice(data_size, data_size)
            rand_dataX = np.array([self.dataX.tolist()[i] for i in rand_ind])
            rand_dataY = np.array([self.dataY.tolist()[i] for i in rand_ind])
            lr.add_evidence(rand_dataX, rand_dataY)
            trainY = lr.query(test_dataX)
            all_Y.append(trainY)
        predY = np.mean(np.array(all_Y), axis=0).tolist()
        return predY
