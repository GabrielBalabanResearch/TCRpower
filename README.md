# About TCRpower
TCRpower is a detection power calculator for determining the number of sequencing reads and number of sampled cells required to detect a T-Cell receptor (TCR) from RNA sequencing.

TCRPower uses a negative binomial model to estimate detection probabilities. This model can be calibrated with your own data which is specific to your experimental set-up. To perform the calibration, you will need to know the ground truth TCR fraction, and sequenced TCR counts for a subset of the TCR in your sequencing library, as well as the total number of reads that were sequenced. An example dataset with a fitted negative binomial model is presented below.

![Example images](/testdata/readcounts_model.png)

Based on the fitted model, you can calculate the probability of detecting a T-Cell in a sample given that you know the true
T-Cell frequency, and the minimum number of reads mapping to a TCR that you require for a T-Cell to be considered "detected".

![Example images](/testdata/powercalc.png)

# How to use the calculator
Download the github repository, and make sure you have the required dependencies. Then run

`python3 setup.py install`

to install the calculator. Check to see if the calculator is working properly by running the automated tests

`python3 unittests.py`

If all of the tests pass then we recommend that you open the Jupyter notebook and try the example calculations.

`example_powercalculations.ipynb`

You can run your own power calculations be either modifying the notebook or creating your own scripts/notebooks 
where you import and use the tcrpower package.

```python
from tcrpower import NBVarCalibrator, TCRPowerCalculator
```

# Known compatible dependencies
* python 3.8.5
* scipy 1.4.1
* numpy 1.17.4
* pandas 1.1.0
* matplotlib 3.1.2
* statsmodels 0.12.1
* numdifftools 0.9.39
* jupyter-notebook 6.1.5

# Citations
See upcoming paper.

<!--

The original crack generation method was developed in 

Costa, Caroline Mendonca, et al. "An efficient finite element approach for modeling fibrotic clefts in the heart." IEEE Transactions on Biomedical Engineering 61.3 (2013): 900-910.

For the extension with topological analysis please cite

Balaban G, et al. "3D Electrophysiological Modeling of InterstitialFibrosis Networks and Their Role in VentricularArrhythmias in Non-ischemic Cardiomyopathy." IEEE Transactions on Biomedical Engineering (upcoming).
DOI 0.1109/TBME.2020.2976924 
 -->

# Lisence 
CC-BY 4.0 or later version (https://creativecommons.org/licenses/)