#!/usr/bin/env python3

import json
import sys

def main():
    f = open(sys.argv[1], "r")
    data = json.load(f)
    pre_name = sys.argv[1].replace(".json", "")
    for key in data.keys():
        print("%s_%s,%f,%f,%f,%e" % (pre_name, key, data[key]["AVG-PR"], data[key]["ROC-AUC"], data[key]["Accuracy"], data[key]["emission"]))
    f.close()

if __name__ in "__main__":
    main()

