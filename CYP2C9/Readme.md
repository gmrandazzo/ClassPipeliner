# CYP2C9 Dataset

Source: [https://pubchem.ncbi.nlm.nih.gov/bioassay/1671198](https://pubchem.ncbi.nlm.nih.gov/bioassay/1671198)

Compound Ranking:

1. Compounds are first classified as having full titration curves, partial modulation, partial curve (weaker actives), single point activity (at highest concentration only), or inactive. See data field "Curve Description". For this assay, apparent inhibitors are ranked higher than compounds that showed apparent activation.

2. For all inactive compounds, PUBCHEM_ACTIVITY_SCORE is 0. For all active compounds, a score range was given for each curve class type given above.
   Active compounds have PUBCHEM_ACTIVITY_SCORE between 40 and 100.
   Inconclusive compounds have PUBCHEM_ACTIVITY_SCORE between 1 and 39.
   Fit_LogAC50 was used for determining relative score and was scaled to each curve class' score range.

In this model we exclude the inconclusive compounds
Active are marked with 1
Inactive are marked with 0

CYP2C9 is known to metabolize 12.8% of 248 clinical used drugs [ref](https://www.sciencedirect.com/science/article/pii/S0163725813000065)


# Results

To reproduce the entire pipeline please execute run.x



