#!/usr/bin/env python3
"""
basic_method.py 

Copyright (C) <2022>  Giuseppe Marco Randazzo <gmrandazzo@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import sys
import os
import random
from copy import copy
from sklearn.linear_model import LogisticRegression
import numpy as np
from codecarbon import track_emissions
from codecarbon import OfflineEmissionsTracker
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from libscientific.pls import PLS

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
import time

class Split():
    """
    Class that define a split train/test/val
    """
    def __init__(self):
        self.x_header = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__


def get_low_variance_descriptors(x_mx):
    """
    Get low variance descriptors
    """
    print(">> Get low variance descriptors")
    skip_ids = []
    for j in range(len(x_mx[0])):
        v = []
        for i in range(len(x_mx)):
            if np.isnan(x_mx[i][j]):
                skip_ids.append(j)
                break
            else:
                v.append(x_mx[i][j])
        try:
            std = np.std(v)
            min_ = np.min(v)
            max_ = np.max(v)
    
            if std > 0:
                if std / (max_ - min_) > 0.01:
                    continue
                else:
                    # skip feature
                    skip_ids.append(j)
            else:
                skip_ids.append(j)
        except:
            skip_ids.append(j)
    return skip_ids


def descriptors_preprocess(x_mx, skip_ids=None):
    """
    Preprocess descriptors removing not relevant columns
    """
    if skip_ids == None:
        """
        Get a list of low variance descriptors and remove them
        """
        skip_lst = get_low_variance_descriptors(x_mx)
    else:
        skip_lst = skip_ids

    print(">> Preprocess descriptors: remove nan and low variance descs.")
    x_pre = []
    for i in range(len(x_mx)):
        x_pre.append(list())
        for j in range(len(x_mx[i])):
            if j in skip_lst:
                continue
            else:
                x_pre[-1].append(x_mx[i][j])
    return np.array(x_pre, dtype=float), skip_lst


def make_sample(xdict, ydict, keys):
    """
    Crate a sample given a key list
    """
    print(">> Create Sample out of a list of keys")
    x_mx = []
    y_mx = []
    if keys is None:
        for key in ydict.keys():
            if key in xdict.keys():
                x_mx.append(xdict[key])
                y_mx.append(ydict[key])
            else:
                continue
    else:
        for key in keys:
            if key in xdict.keys() and key in ydict.keys():
                x_mx.append(xdict[key])
                y_mx.append(ydict[key])
            else:
                continue
    
    return np.array(x_mx, dtype=float), np.array(y_mx, dtype=int)


def make_split(xdict : dict,
               x_header : list,
               ydict : dict,
               random_state=2785):
    """
    Create a split of train/test/val to compare all classifiers.
    """
    print(">> Create a random Train/Test/Validation split")
    split = Split()
    keys = list(ydict.keys())
    k_sub, k_val = train_test_split(keys,
                                    test_size=0.33,
                                    random_state=random_state)
    k_train, k_test = train_test_split(k_sub,
                                       test_size=0.2,
                                       random_state=random_state)
    
    split.x_train, split.y_train = make_sample(xdict, ydict, k_train)
    split.x_test, split.y_test = make_sample(xdict, ydict, k_test)
    split.x_val, split.y_val = make_sample(xdict, ydict, k_val)
    
    """
    Preprocess descriptors by removing the less important descriptors
    """
    x_all, _ = make_sample(xdict, ydict, None)
    x_all_pre, skip_ids = descriptors_preprocess(x_all, None)    
    split.x_train, _ = descriptors_preprocess(split.x_train, skip_ids)
    split.x_test, _ = descriptors_preprocess(split.x_test, skip_ids)
    split.x_val, _ = descriptors_preprocess(split.x_val, skip_ids)
    split.x_header = []
    for i in range(len(x_header)):
        if i in skip_ids:
            continue
        else:
            split.x_header.append(x_header[i])
    return split


def make_balanced_split(xdict : dict,
                        x_header : list,
                        ydict : dict,
                        random_state=2785):
    """
    Create a split of train/test/val to compare all classifiers.
    Sample all 0 and all 1
    Split equally 0 and 1 and add them to train/test/val
    """
    print(">> Create a balanced Train/Test/Validation split")
    split = Split()
    zero = []
    one = []
    for key in ydict.keys():
        if int(ydict[key])  == 0:
            zero.append(key)
        else:
            one.append(key)

    z_k_sub, z_k_val = train_test_split(zero,
                                        test_size=0.33,
                                        random_state=random_state)
    z_k_train, z_k_test = train_test_split(z_k_sub,
                                           test_size=0.2,
                                           random_state=random_state)

    o_k_sub, o_k_val = train_test_split(one,
                                        test_size=0.33,
                                        random_state=random_state)
    o_k_train, o_k_test = train_test_split(o_k_sub,
                                           test_size=0.2,
                                           random_state=random_state)

    k_train = z_k_train + o_k_train
    k_test = z_k_test + o_k_test
    k_val = z_k_val + o_k_val
    split.x_train, split.y_train = make_sample(xdict, ydict, k_train)
    split.x_test, split.y_test = make_sample(xdict, ydict, k_test)
    split.x_val, split.y_val = make_sample(xdict, ydict, k_val)
    
    """
    Preprocess descriptors by removing the less important descriptors
    """
    x_all, _ = make_sample(xdict, ydict, None)
    x_all_pre, skip_ids = descriptors_preprocess(x_all, None)    
    split.x_train, _ = descriptors_preprocess(split.x_train, skip_ids)
    split.x_test, _ = descriptors_preprocess(split.x_test, skip_ids)
    split.x_val, _ = descriptors_preprocess(split.x_val, skip_ids)
    split.x_header = []
    for i in range(len(x_header)):
        if i in skip_ids:
            continue
        else:
            split.x_header.append(x_header[i])
    return split

def tf_dnn(split : Split):
    """
    Create a simple feed forward neural network
    """
    print(">> Compute DNN with tensorflow")
    nfeatures = split.x_train.shape[1]
    """
    TUNING PARAMETERS
    """
    nunits = 256
    ndense_layers = 2
    epochs_= 300
    batch_size_ = 20

    model = Sequential()
    model.add(BatchNormalization(input_shape=(nfeatures,)))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dropout(0.15))
    for i in range(ndense_layers):
        model.add(Dense(nunits, activation='relu'))
        model.add(Dropout(0.1))
    model.add(Dense(nunits, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    pr_auc = AUC(num_thresholds=200,
                 curve='PR',
                 summation_method='interpolation')
    
    """
    Binary cross entropy because is a 2 class classification problem
    Look always to PR and not accuracy for unbalanced classification problems
    """
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.00005),
                  metrics=['accuracy', pr_auc])
 
    log_dir_ = ('./dnnlogs/')
    log_dir_ += time.strftime('%Y%m%d%H%M%S')
    model_output = "dnnmodel.h5"
    """
    Use model checkpoints to save the best predictive model
    """
    callbacks_ = [TensorBoard(log_dir=log_dir_,
                                histogram_freq=0,
                                write_graph=False,
                                write_images=False),
                 ModelCheckpoint(model_output,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True)]


    model.fit(split.x_train, split.y_train,
              epochs=epochs_,
              batch_size=batch_size_,
              verbose=0,
              validation_data=(split.x_test, split.y_test),
              callbacks=callbacks_)
    """
    Load the best model and predict the validation set
    """
    bestmodel = load_model(model_output)
    y_val_pred = bestmodel.predict(split.x_val)
    os.remove(model_output)
    del model
    del pr_auc
    return y_val_pred

def pls_da(split : Split):
    clf = PLS(nlv=6, xscaling=1, yscaling=0)
    y_train = [[item] for item in split.y_train]
    clf.fit(split.x_train, y_train)
    p_c_y_test, _ = clf.predict(split.x_test)
    # Select the best latent variables
    res = []
    for c in range(len(p_c_y_test[0])):
        ypred = []
        for i in range(len(p_c_y_test)):
            ypred.append(p_c_y_test[i][c])
        res.append(average_precision_score(split.y_test, np.array(ypred)>0.5))    
    c = 0
    for i in range(1, len(res)):
        if res[i] > res[i-1] and np.abs(res[i]-res[i-1]+res[i-1]) > 0.01:
            c += 1
        else:
            break
    c = np.argmax(res)
    p_c_y_val, _ = clf.predict(split.x_val)
    p_y_val = []
    for i in range(len(p_c_y_val)):
        p_y_val.append(p_c_y_val[i][c])
    return p_y_val


def classify(split : Split):
    """
    Classify using sklearn and tensorflow models
    """
    print(">> Classify ")
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]


    """
    In this case we do not use the test set for algorithm reasons.
    """
    
    class_results = {}
    emissions_results = {}
    for name, clf in zip(names, classifiers):
        print(" * Calculating %s" % (name))
        tracker = OfflineEmissionsTracker(country_iso_code="CHE")
        tracker.start()
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(split.x_train, split.y_train)
        class_results[name] = clf.predict(split.x_val)
        emissions_results[name] = float(tracker.stop())
    
    print(" * Calculating PLS-DA")
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    class_results["PLS-DA"] = pls_da(split)
    emissions_results["PLS-DA"] = float(tracker.stop())
    
    print(" * Calculating DNN")
    tracker = OfflineEmissionsTracker(country_iso_code="CHE")
    tracker.start()
    class_results["DNN"] = tf_dnn(split)
    emissions_results["DNN"] = float(tracker.stop())
    return class_results, emissions_results


def elaborate_results(y_true, class_results : dict, emissions_results):
    """
    Elaborate the results in therms of precision recall auc.
    The higher the best it is.
    """
    print(">> Elaboerate the results...")
    res = {}
    for key in class_results.keys():
        y_score = class_results[key]
        res[key] = {}
        res[key]["AVG-PR"] = average_precision_score(y_true, y_score)
        res[key]["ROC-AUC"] = roc_auc_score(y_true, y_score)
        res[key]["Accuracy"] = accuracy_score(y_true, np.array(y_score)>0.5)
        res[key]["emission"] = emissions_results[key]
    return res


def best_classifier(xdict : dict, x_header : list, ydict : dict):
    """
    Find the best classifier
    """
    print(">> Search for best classifier")
    # split = make_split(xdict, x_header, ydict, 2785) << random split!!!
    split = make_balanced_split(xdict, x_header, ydict, 2785) # better split
    class_results, emissions_results = classify(split)
    
    return elaborate_results(split.y_val, class_results, emissions_results), split

def write_results(res, out_name):
    fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    x_emission = []
    y_avgpr = []
    y_rocauc = []
    y_accuracy = []
    method_name = []
    for key in res.keys():
        method_name.append(key)
        x_emission.append(res[key]["emission"])
        y_avgpr.append(res[key]["AVG-PR"])
        y_rocauc.append(res[key]["ROC-AUC"])
        y_accuracy.append(res[key]["Accuracy"])
        
    axs[0, 0].scatter(x_emission, y_avgpr, s=80, c="blue", marker="o")
    axs[0, 0].set_ylabel("Average PR")
    axs[0, 0].set_xlabel("Emission")
    
    size = 6
    for i, txt in enumerate(method_name):
        axs[0, 0].annotate(txt, (x_emission[i], y_avgpr[i]), fontsize=size)
    
    axs[0, 1].scatter(x_emission, y_rocauc, s=80, c="black", marker="o")
    axs[0, 1].set_ylabel("ROC AUC")
    axs[0, 1].set_xlabel("Emission")
    
    for i, txt in enumerate(method_name):
        axs[0, 1].annotate(txt, (x_emission[i], y_rocauc[i]), fontsize=size)
    
    axs[1, 0].scatter(x_emission, y_accuracy, s=80, c="green", marker="o")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].set_xlabel("Emission")

    for i, txt in enumerate(method_name):
        axs[1, 0].annotate(txt, (x_emission[i], y_accuracy[i]), fontsize=size)

    axs[1, 1].barh(method_name, y_avgpr)
    axs[1, 1].set_ylabel("ML Method")
    axs[1, 1].set_xlabel("Average PR")


    plt.tight_layout()
    plt.savefig("%s.png" % (out_name), dpi=300)
    with open("%s.json" % (out_name), "w") as write_file:
        json.dump(res, write_file, indent=4)
    return

def write_html_barplot_output(res, html_out):
    print(">> Write the final result")
    f_html = open(html_out, "w",  encoding="utf-8")
    f_html.write('<!DOCTYPE html>\n')
    f_html.write('<html lang="en">\n')
    f_html.write('<head>\n')
    f_html.write('  <!-- Load plotly.js into the DOM -->\n')
    f_html.write('  <script src="https://cdn.plot.ly/plotly-2.16.1.min.js"></script>\n')
    f_html.write('</head>\n')
    f_html.write('<body>\n')
    f_html.write('  <div id="myDiv"></div>\n')
    f_html.write('<script>\n')
    
    xrow = ""
    yrow = ""
    for key in res.keys():
        xrow += "'%s'," % (key)
        yrow += "%f," % (res[key]["Avg-PR"])
        
    f_html.write('var data = [\n')
    f_html.write('  {\n')
    f_html.write('    x: [%s],\n' % (xrow))
    f_html.write('    y: [%s],\n' % (yrow))
    f_html.write('    type: "bar",\n')
    f_html.write('  }\n')
    f_html.write('];\n')

    f_html.write('Plotly.newPlot("myDiv", data);\n')
    f_html.write('</script>\n')
    f_html.write('</body>\n')
    f_html.close()

def write_html_variable_importance(res):
    method_keys = []
    first_key = list(res.keys())[0]
    for key in res[first_key].keys():
        method_keys.append(key)

    
    for keym in method_keys:
        v_imp = {}
        for key in res.keys():
            v_imp[key] = res[key][keym]
        write_html_barplot_output(v_imp, "%s.html" % (keym))
    
def variable_importance(split : Split,
                        res_all_vars : dict):
    """
    Variable importance
    This take times...
    We build N models with N the number of variable and we see how
    the performance decrease while killing a varialbe.
    We define a score S which represent the variable importance
    as s_current/s_original.
    Values > 1 means killing this variable is better
    Values < 1 means this variable is important to explain the model
    """
    print(">> Calculate the varible importance")
    v_imp = {}
    for j in range(len(split.x_header)):
        var_name = split.x_header[j]
        print("Kill %s" % (var_name))
        split_copy = copy(split)
        print(split_copy.x_train.shape)
        split_copy.x_train = np.delete(split_copy.x_train, j, 1)
        split_copy.x_test = np.delete(split_copy.x_test, j, 1)
        split_copy.x_val = np.delete(split_copy.x_val, j, 1)
        class_results = classify(split_copy)
        curr_res = elaborate_results(split_copy.y_val, class_results)
        v_imp[var_name] = {}
        for key in res_all_vars.keys():
            v_imp[var_name][key] = curr_res[key]/res_all_vars[key]  
    return v_imp
