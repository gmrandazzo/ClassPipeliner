# Acquatic Ecotoxicity

Source: [https://pubs.rsc.org/en/content/articlelanding/2016/gc/c5gc02818c](https://pubs.rsc.org/en/content/articlelanding/2016/gc/c5gc02818c)


EC50: Concentration of a drug at which 50% of its maximal response is induced. 
      The lower the EC50 value, the lower the concentration of drug required to 
      show a 50% of activity. In this case the lower is the more toxic it is.


# Results

Classification models EC50 < 0.001

1: Toxic
0: Non toxic

DNN (neural network) requires a lot of energy compared to the other ML methods.
On D. Magna, DNN performs way better than the other methods.

On D. Promelas, instead, a simpler method, such as the Partial Least Squares Discriminant analysis (PLS-DA)
perform similarly and require 200 less energy to develop an ML prediction model.

On average, a DNN requires 0.02 Watt; PLS-DA requires 0.0001 Watt.


## D. Magna classification results
![D. Magna classification results](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/AcquaticEcoToxicity/d_magna.fp.simple.png)

## D. Promelas classification results
![D. Promelas classification results](https://raw.githubusercontent.com/gmrandazzo/ClassPipeliner/main/AcquaticEcoToxicity/d_promelas.fp.simple.png)

