# CYP2C19 Dataset

Source: [https://pubchem.ncbi.nlm.nih.gov/bioassay/1671197](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671197)

Compound Ranking:

1. Compounds are first classified as having full titration curves, partial modulation, partial curve (weaker actives), single point activity (at highest concentration only), or inactive. See data field "Curve Description". For this assay, apparent inhibitors are ranked higher than compounds that showed apparent activation.

2. For all inactive compounds, PUBCHEM_ACTIVITY_SCORE is 0. For all active compounds, a score range was given for each curve class type given above.
   Active compounds have PUBCHEM_ACTIVITY_SCORE between 40 and 100.
   Inconclusive compounds have PUBCHEM_ACTIVITY_SCORE between 1 and 39.
   Fit_LogAC50 was used for determining relative score and was scaled to each curve class' score range.

In this model we exclude the inconclusive compounds
Active are marked with 1
Inactive are marked with 0

CYP2C19 is known to metabolize 6.8% of 248 clinical used drugs [ref](https://www.sciencedirect.com/science/article/pii/S0163725813000065)


# Results

To reproduce the entire pipeline please execute run.x

## Premise

These results are calculated using two molecular representations:

* ECFP fingerprint
* RDKit molecular descriptors

Train/test/validation sets are equal at each run of the ML method and thus comparable. 

For each model we calculate as metrics:
- area under the precision-recall curve (AVG PR)
- area under the receiver operating characteristic curve (ROC AUC)
- accuracy
- emission (kW): the energy necessary to train the model.

These scores are calculated using ONLY the validation set. Hence, no train/test values are considered.

## To the results

Molecular descriptors perform better than ECFP.
DNN model shows an AVG PR of 0.7, which starts becoming interesting. However, the model is still a bad classifier.
Hence, our CYP2C19 model can be used for some inference. However, we need to be cautious!
Moreover, a simpler method, such as the Partial Least Squares Discriminant analysis (PLS-DA)
achieve similar results to DNN on the validation set, requiring 110 less energy to develop an ML prediction model.

In this case a DNN requires 0.11 Watt; PLS-DA requires 0.001 Watt


| ECFP RESULTS  | DESC RESULTS  |
| ------------- |:-------------:|
| ![KCGS Results11](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/CYP2C19/dataset.morgan_ecfp.png) | ![KCGS Results12](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/CYP2C19/dataset.rdkit_dscriptors.png)     |


