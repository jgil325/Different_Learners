""""""
""" 
Test a learner.  

Copyright 2018-2022
Georgia Institute of Technology (Georgia Tech) 
Atlanta, Georgia 30332 
All Rights Reserved 

Copyright 2018-2022
University of Georgia (UGA)
Athens, GA            
All Rights Reserved   
 
Template code for CSCI [4|6]170 Computational Investments
 
Georgia Tech & University of Georgia asserts copyright ownership of this template 
and all derivative works, including solutions to the projects assigned in this 
course. 

Students and other users of this template code are advised not to share it 
with others or to make it available on publicly viewable websites including 
repositories such as github and gitlab.  This copyright statement should not be 
removed or edited. 
 
We do grant permission to share solutions privately with non-students such 
as potential employers. However, sharing with other current or future 
students of CS 7646 (GTech) or CSCI [4|6]170  (UGA)  is prohibited and subject 
to being investigated as a GT honor code violation or/and UGA honor code 
violation.
 
-----do not edit anything above this line--- 
"""

import math
import sys

import numpy as np

import LinRegLearner as lrl
import DTLearner as dt
import BagLearner as bl
import RTLearner as rt
import InsaneLearner as it

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        # sys.exit(1)
        inf = open('Data/ripple.csv')
    else:
        inf = open(sys.argv[1])
    data = np.array(
        [list(map(float, s.strip().split(","))) for s in inf.readlines()]
    )

    # compute how much of the data is training and testing 
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data 
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # -----------LIN REG LEARNER---------------------------------------------------------------------------------------
    print("-----------LIN REG LEARNER------------")
    # create a learner and train it 
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner 
    learner.add_evidence(train_x, train_y)  # train it 
    print(learner.author())

    # evaluate in sample 
    pred_y = learner.query(train_x)  # get the predictions 
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample 
    pred_y = learner.query(test_x)  # get the predictions 
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    # -----------DT LEARNER-----------------------------------------------------------------------------------------
    print("-----------DT LEARNER------------")
    # create a learner and train it
    test_dt_learner = dt.DTLearner(leaf_size=1, verbose=False)  # constructor
    test_dt_learner.add_evidence(train_x, train_y)  # train it
    print(test_dt_learner.author())

    # evaluate in sample
    pred_y = test_dt_learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = test_dt_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    # -----------RT LEARNER-----------------------------------------------------------------------------------------
    print("-----------RT LEARNER------------")
    # create a learner and train it
    test_rt_learner = rt.RTLearner(leaf_size=1, verbose=False)  # constructor
    test_rt_learner.add_evidence(train_x, train_y)  # train it
    print(test_rt_learner.author())

    # evaluate in sample
    pred_y = test_rt_learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = test_rt_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    # -----------BAG LEARNER 1-----------------------------------------------------------------------------------------
    print("-----------BAG LEARNER 1------------")
    # create a learner and train it
    test_bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
    test_bag_learner.add_evidence(train_x, train_y)  # train it
    print(test_bag_learner.author())

    # evaluate in sample
    pred_y = test_bag_learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = test_bag_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    # -----------BAG LEARNER 2-----------------------------------------------------------------------------------------
    print("-----------BAG LEARNER 2------------")
    # create a learner and train it
    test_bag_learner = bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=10, boost=False, verbose=False)
    test_bag_learner.add_evidence(train_x, train_y)  # train it
    print(test_bag_learner.author())

    # evaluate in sample
    pred_y = test_bag_learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = test_bag_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")

    # -----------INSANE LEARNER-----------------------------------------------------------------------------------------
    print("-----------INSANE LEARNER------------")
    # create a learner and train it
    test_ins_learner = it.InsaneLearner(verbose = False) # constructor
    test_ins_learner.add_evidence(train_x, train_y)  # train it
    print(test_ins_learner.author())

    # evaluate in sample
    pred_y = test_ins_learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = test_ins_learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0, 1]}")
