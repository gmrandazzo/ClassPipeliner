# CYP3A4 Dataset

Source: [https://pubchem.ncbi.nlm.nih.gov/bioassay/1671201](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671201)

Compound Ranking:

1. Compounds are first classified as having full titration curves, partial modulation, partial curve (weaker actives), single point activity (at highest concentration only), or inactive. See data field "Curve Description". For this assay, apparent inhibitors are ranked higher than compounds that showed apparent activation.

2. For all inactive compounds, PUBCHEM_ACTIVITY_SCORE is 0. For all active compounds, a score range was given for each curve class type given above.
   Active compounds have PUBCHEM_ACTIVITY_SCORE between 40 and 100.
   Inconclusive compounds have PUBCHEM_ACTIVITY_SCORE between 1 and 39.
   Fit_LogAC50 was used for determining relative score and was scaled to each curve class' score range.

In this model we exclude the inconclusive compounds
Active are marked with 1
Inactive are marked with 0

CYP3A4 is known to metabolize 30% of 248 clinical used drugs [ref](https://www.sciencedirect.com/science/article/pii/S0163725813000065)

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
Most of the models generated here are poor classifiers, even if ROC AUC says 0.8 for some ML methods.
A random classifier has an AUC PR near 0.5. In this case, some models show 0.6 or nearby.
With that, our CYP3A4 model can be used for some inference. However, we need to be cautious!

| ECFP RESULTS  | DESC RESULTS  |
| ------------- |:-------------:|
| ![KCGS Results11](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/CYP3A4/dataset.morgan_ecfp.png) | ![KCGS Results12](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/CYP3A4/dataset.rdkit_dscriptors.png)     |

