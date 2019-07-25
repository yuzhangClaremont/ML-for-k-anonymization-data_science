import pandas as pd
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

def decisionTree(parameter, df, measurement):
    train, test = train_test_split(df, test_size = 0.15)
    # suppressed data set numeric encoding

    # Create a label encoder object
    le = LabelEncoder()

    # Iterate through the columns
    for col in train:
        if train[col].dtype == 'object':
            le.fit(suppressed_df[col])
            # Transform both training and testing data
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
    
    train_results = []
    train_f1 = []
    test_results = []
    test_f1 = []
    if parameter = "max_depth":
        max_depths = np.linspace(1, 32, 32, endpoint=True)
        for max_depth in max_depths:

            dt = DecisionTreeClassifier(max_depth=max_depth)
            dt.fit(x_train, y_train)
            train_pred = dt.predict(x_train)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            train_results.append(roc_auc)
            
            p, r, f1, s = metrics.precision_recall_fscore_support(y_train, train_pred,
                                                            average="weighted")
            train_f1.append(f1)
            # Add auc score to previous train results
            
            y_pred = dt.predict(x_test)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)

            p, r, f1, s = metrics.precision_recall_fscore_support(y_test, y_pred,
                                                            average="weighted")
            # Add auc score to previous test results
            test_results.append(roc_auc)
            test_f1.append(f1)
            # print(test_f1)
            # print(train_f1)
        from matplotlib.legend_handler import HandlerLine2D
        line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
        line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('AUC score')
        plt.xlabel('Tree depth')
        plt.show() # 8 max_depth = 8


def main():
        # take in and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_dir", "-i", type=str, required=True, help='path to directory of input dataset')
    parser.add_argument("--MLmodel", "-l", type=str, required=True, help='the machine learning model to tune: dt, logReg')
    parser.add_argument("--parameter", "-p", type=str, required = True, help='parameter to tune: max_depth')
    parser.add_argument("--measurement", "-m", type=str, required = True, help='measurement to tune: f1, auc')
    parser.add_argument("--visualize", "-v", action='store_true', help='flag for visualizations')
    
    input_df = args["input_dataset_dir"]
    MLmodel = args["MLmodel"]
    parameter = args["parameter"]
    measurement = args["measurement"]
    vis = args["vis"]

    df = pd.read_csv()
if __name__ == "__main__":
    main()