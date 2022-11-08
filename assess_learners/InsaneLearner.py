import numpy as np
import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner(object):
    def __init__(self, verbose=False):
        pass

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
        learner_bl = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False)
        bag_learners = []
        for i in range(0, 20):
            bag_learners.append(learner_bl)
        for lr in bag_learners:
            lr.add_evidence(self.dataX, self.dataY)
            trainY = lr.query(test_dataX)
            all_Y.append(trainY)
        predY = np.mean(np.array(all_Y), axis=0).tolist()
        return predY
