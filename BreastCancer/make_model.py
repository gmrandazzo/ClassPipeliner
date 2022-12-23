#!/usr/bin/env python3
"""
make_model.py 

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

Algorhtm:

Run several classifiers also a neural network and get the best model

   
"""

import sys
import os
from copy import copy
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (dir_path))
from basic_methods import best_classifier
from basic_methods import write_results
from basic_methods import write_html_barplot_output
from sklearn.datasets import load_breast_cancer

def load_dataset():
    dataset = load_breast_cancer(as_frame=True)
    x_dict = {}
    x_header = list(dataset['data'].columns)
    y_dict = {}
    for i in range(len(dataset['data'].values)):
        x_dict["Obj%d" % (i)] = dataset['data'].values[i]
        y_dict["Obj%d" % (i)] = dataset['target'].values[i]
    return x_dict, x_header, y_dict


def main():
    if len(sys.argv) != 1:
        print("Usage %s " % (sys.argv[0]))
    else:
        x_dict, x_header, y_dict = load_dataset()
        res, _ = best_classifier(x_dict, x_header, y_dict)
        outname = "breast_cancer_best_classifier.html"
        #write_html_barplot_output(res, outname)
        write_results(res, "breast_cancer_results")
    return

if __name__ in "__main__":
    main()
