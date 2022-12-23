# Kinase Chemogenomic Set

Source: [https://www.mdpi.com/1422-0067/22/2/566](https://www.mdpi.com/1422-0067/22/2/566)

Every target represents the %inhibition at 1 uM of substrate concentration.
These values are acquired using the [KINOMEscan technology](https://www.discoverx.com/technologies-platforms/competitive-binding-technology/kinomescan-technology-platform).
Every target value is converted into a classification problem.
We consider "Binder" a compound with a %inhibition > 70% (1 value)
We consider "Less binder" a compound with a %inhibition < 70% (0 value)

This model aims to estimate if a compound may bind or not a kinase more than 70%.

# Results

To reproduce the entire pipeline please execute run.x

## Premise

These results are calculated using two molecular representations:
* ECFP fingerprint
* RDKit molecular descriptors

Train/test/validation sets are equal at each run of the ML method and thus comparable. 

For each model on each kinase target we calculate as metrics:
- area under the precision-recall curve (AVG PR)
- area under the receiver operating characteristic curve (ROC AUC)
- accuracy
- emission (kW): the energy necessary to train the model.


These scores are calculated using ONLY the validation set. Hence, no train/test values are considered.

## To the results

Most of the model generated here are poor classifier even if ROC AUC says 0.8.
A random classifier has an AUC PR near by 0.5. In our case we are always down 0.5. 
With that, our models do not perform better than a random selection. Sad but true!
Some kinase target instead shows good AUC PR > 0.6. Those ones can be used for inference.
In general, ECFP models perform better than descriptors.

### ECFP results

|               |               |
| ------------- |:-------------:|
| ![KCGS Results1](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/AdaBoost-rdkit_ecfp_table_results_by_kinase.png)     | ![KCGS Results6](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Gaussian_Process-rdkit_ecfp_table_results_by_kinase.png)     |
| ![KCGS Results2](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Linear_SVM-rdkit_ecfp_table_results_by_kinase.png)      | ![KCGS Results7](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/RBF_SVM-rdkit_ecfp_table_results_by_kinase.png)     |
| ![KCGS Results3](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Naive_Bayes-rdkit_ecfp_table_results_by_kinase.png)      | ![KCGS Results8](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/QDA-rdkit_ecfp_table_results_by_kinase.png)     |
| ![KCGS Results4](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Nearest_Neighbors-rdkit_ecfp_table_results_by_kinase.png)      | ![KCGS Results9](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Random_Forest-rdkit_ecfp_table_results_by_kinase.png)     |
| ![KCGS Results5](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/DNN-rdkit_ecfp_table_results_by_kinase.png)      | ![KCGS Results10](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/PLS-DA-rdkit_ecfp_table_results_by_kinase.png)    |


### Descriptors results

|               |               |
| ------------- |:-------------:|
| ![KCGS Results1](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/AdaBoost-rdkit_desc_table_results.png)      | ![KCGS Results6](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Gaussian_Process-rdkit_desc_table_results.png)     |
| ![KCGS Results2](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Linear_SVM-rdkit_desc_table_results.png)      | ![KCGS Results7](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/RBF_SVM-rdkit_desc_table_results.png)     |
| ![KCGS Results3](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Naive_Bayes-rdkit_desc_table_results.png)      | ![KCGS Results8](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/QDA-rdkit_desc_table_results.png)     |
| ![KCGS Results4](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Nearest_Neighbors-rdkit_desc_table_results.png)      | ![KCGS Results9](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/Random_Forest-rdkit_desc_table_results.png)     |
| ![KCGS Results5](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/DNN-rdkit_desc_table_results.png)      | ![KCGS Results10](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/PLS-DA-rdkit_desc_table_results.png)     |


### Carbon footprint analysis

These results show the average precision-recall area (AVG PR) versus the receiver operating characteristic area (ROC AUC) for all kinases targets per ML method.
The size of every spot represents the energy "impact" utilized to train the ML model.

| ECFP RESULTS  | DESC RESULTS  |
| ------------- |:-------------:|
| ![KCGS Results11](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/rdkit_ecfp_table_results.png) | ![KCGS Results12](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/rdkit_desc_table_results.png)     |

DNN (neural network) in both molecular descriptions requires a lot of energy compared to the other ML methods.
Moreover, a simpler method, such as the Partial Least Squares Discriminant analysis (PLS-DA)
perform better on the validation prediction and require 200 less energy to develop an ML prediction model.

On average, a DNN requires 0.02 Watt; PLS-DA requires 0.0001 Watt
Since we have trained 252 models, we have used 5 Watts to train the DNN and only 0.025 Watts to train PLS-DA for the same dataset and get a similar answer.

Detailed results about the external validation set performances per kinase target 
can be consulted on the respective directories [RDKitDescriptorsResults](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/RDKitDescriptorsResults) and [RDKitECFPMorganFP](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/KinaseChemoGenomicSet/FinalResults/RDKitECFPMorganFP).



# Interesting articles

- [Deep Learning Enhancing Kinome-Wide Polypharmacology Profiling: Model Construction and Experiment Validation](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00855)

- [Global Analysis of Deep Learning Prediction Using Large-Scale In-House Kinome-Wide Profiling Data](https://pubs.acs.org/doi/10.1021/acsomega.2c00664)
