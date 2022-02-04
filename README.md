# About TCRpower
TCRpower is a detection power calculator for determining the number of sequencing reads and number of sampled cells required to detect a T-Cell receptor (TCR) from RNA sequencing.

TCRPower uses a negative binomial model to estimate detection probabilities. This model can be calibrated with your own data which is specific to your experimental set-up. To perform the calibration, you will need to know the ground truth TCR fraction, and sequenced TCR counts for a subset of the TCR in your sequencing library, as well as the total number of reads that were sequenced. An example dataset with a fitted negative binomial model is presented below.

![Example images](/testdata/readcounts_model.png)

Based on the fitted model, you can calculate the probability of detecting a T-Cell in a sample given that you know the true T-Cell frequency, and the minimum number of reads mapping to a TCR that you require for a T-Cell to be considered "detected".

![Example images](/testdata/powercalc.png)

# How to use the calculator
Download the github repository, and make sure you have the dependent packages installed.

* python (3.8.5)
* scipy (1.4.1)
* numpy (1.17.4)
* pandas (1.1.0)
* matplotlib (3.1.2)
* statsmodels (0.12.1)
* numdifftools (0.9.39)
* jupyter-notebook (6.1.5)

The package version numbers in brackets are known to be compatible with TCRPower, but other versions may work as well. 
If you would like to keep these packages seperate from the rest of your system you can use a package manager like [Miniconda](https://docs.conda.io/) to
run TCRPower in a virtual environment.

Once you have the dependencies you can run

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

# Citations
If you found TCRPower useful in your research please cite the [TCR paper](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbab566/6513728) which was published in the journal Briefings in Bioinformatics.

# Lisence 
[CC-BY 4.0(https://creativecommons.org/licenses/) or later version.