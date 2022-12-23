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
        kname = []
        for key in d.keys():
            if m in key:
                avpr.append(d[key][0])
                rocauc.append(d[key][1])
                acc.append(d[key][2])
                emis.append(d[key][3])
                name = key.split("/")[1].replace(m, "")
                kname.append(name)
            else:
                continue
        avpr = np.array(avpr).astype(float)
        rocauc = np.array(rocauc).astype(float)
        acc = np.array(acc).astype(float)
        emis = np.array(emis).astype(float)
        res[m] = {"AUC PR": avpr, "ROC AUC": rocauc, "Accuracy": acc, "Emissions": emis, "names": kname}
    return res

def plot(res, outfig):
    for m in res.keys():
        fig, ax = plt.subplots()
        ids = range(len(res[m]["names"]))
        bars = ax.barh(ids, res[m]["AUC PR"])
        labels = []
        for i, v in enumerate(res[m]["AUC PR"]):
            if v > 0.65:
                labels.append(res[m]["names"][i])
            else:
                labels.append("")
        ax.bar_label(bars, labels=labels, padding=8, color='b', fontsize=6)
        ax.set_ylabel("Kinase")
        ax.set_xlabel("AUC PR")
        ax.set_title("%s" % (m))
        ax.set_xlim([0,1])
        plt.savefig("%s-%s" % (m.replace(" ", "_"), outfig), dpi=300)


def main():
    res1=process(sys.argv[1])
    plot(res1, sys.argv[2])

if __name__ in "__main__":
    main()

