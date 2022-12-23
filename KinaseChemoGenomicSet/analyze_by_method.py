#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def readinp(finp):
    d = {}
    f = open(finp, "r")
    for line in f:
        v = str.split(line.strip(), ",")
        d[v[0]] = v[1:]
    f.close()
    return d

def process(finp):
    d = readinp(finp)
    methods=["PLS-DA", "DNN", "Linear SVM", "Nearest Neighbors", "RBF SVM", "Gaussian Process", "Random Forest", "AdaBoost", "Naive Bayes", "QDA"]
    res = {}
    for m in methods:
        avpr = []
        rocauc = []
        acc=[]
        emis=[]
        for key in d.keys():
            if m in key:
                avpr.append(d[key][0])
                rocauc.append(d[key][1])
                acc.append(d[key][2])
                emis.append(d[key][3])
            else:
                continue
        avpr = np.array(avpr).astype(float)
        rocauc = np.array(rocauc).astype(float)
        acc = np.array(acc).astype(float)
        emis = np.array(emis).astype(float)
        res[m] = {"AUC PR": avpr.mean(), "ROC AUC": rocauc.mean(), "Accuracy": acc.mean(), "Emissions": emis.mean()}
    return res

def plot(res, outfig, title):
    x = []
    y = []
    methods = list(res.keys())
    np.random.seed(1234)
    c = np.random.rand(len(methods))
    area = []
    for key in methods:
        x.append(res[key]["ROC AUC"])
        y.append(res[key]["AUC PR"])
        area.append((5e7*res[key]["Emissions"]))
    plt.scatter(x, y, s=area, c=c, alpha=0.5)
    plt.title(title)
    plt.xlabel("ROC AUC")
    plt.ylabel("AVG PR")
    size = 6
    for i, txt in enumerate(methods):
       plt.annotate(txt, (x[i], y[i]), fontsize=size)
    plt.savefig(outfig, dpi=300)


def main():
    res1=process(sys.argv[1])
    plot(res1, sys.argv[2], sys.argv[3])

if __name__ in "__main__":
    main()

