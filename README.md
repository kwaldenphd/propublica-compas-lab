# ProPublica Compas Lab

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>
This tutorial is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Lab Goals

## Acknowledgements

This lab is based on the research and technical documentation for *ProPublica*'s 2016 "Machine Bias" article.
- Julia Angwin, Jeff Larson, Surya Mtatu, and Lauren Kirchner, “[Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)” *ProPublica* (23 May 2016). 
- Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin, “[How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)” *ProPublica* (23 May 2016).
- [Northpointe document collection gathered by ProPublica team](https://www.documentcloud.org/public/search/%22Project%20ID%22:%20%2227022-compas-documents%22)
- [Sentencing reports that include risk assessment, gathered by ProPublica team](https://www.documentcloud.org/public/search/%22Project%20ID%22:%20%2224665-sample-psis-that-include-risk-assessments%22)
- [GitHub repository with data files and Jupyter notebook for ProPublica analysis of COMPAS risk scores](https://github.com/propublica/compas-analysis)

The lab is also adapted from a lab developed by [Lauren F. Klein](https://lklein.com/) for the Spring 2020 Emory University course [QTM 490 "Feminist Data Science"](https://github.com/laurenfklein/feminist-data-science).
- [Lab activity resources (data + Jupyter Notebook)](https://github.com/laurenfklein/feminist-data-science/tree/master/notebooks/lab3-compas)

Klein's lab is based on an exercise by Bucknell University Assistant Professor of Computer Science [Darakhshan Mir](http://eg.bucknell.edu/~djm056/).

# Table of Contents



# Overview

In 2016, a *ProPublica* investigative journalism team including published "Machine Bias," an incisive look at the COMPAS risk prediction system used in the criminal justice system. In addition to the main story that emphasizes the human toll of a racially-biased system, the *ProPublica* team published a detailed methodology, data, and technical documentation for the story.

This lab is based on that work.

# The Story

Read: Julia Angwin, Jeff Larson, Surya Mtatu, and Lauren Kirchner, “[Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)” *ProPublica* (23 May 2016). 

Discussion questions: 
- What stood out to you from the story? 
  * Where and how do we see race and surveillance at work?
- From reading the story, what are you able to tell about how the authors' analyzed the technology system?
- How do the authors describe the role/functionality/design/etc. of the technology system?
- Other thoughts/comments/questions/observations.

# The methodology

Read: Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin, “[How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)” *ProPublica* (23 May 2016).

Discussion questions:
- What was the goal or central questions for the study?
- What data sources did they draw on (and why)?
- How did they approach analyzing the algorithm? 
  * Specific questions
  * Statistical methods
  * Visualizations/calculations
- What conclusions do the authors draw from the analysis?
  * Connections we can make with race and surveillance?
- Other thoughts/comments/questions/observations.

# The data

Explore:
- [Northpointe document collection gathered by ProPublica team](https://www.documentcloud.org/public/search/%22Project%20ID%22:%20%2227022-compas-documents%22)
- [Sentencing reports that include risk assessment, gathered by ProPublica team](https://www.documentcloud.org/public/search/%22Project%20ID%22:%20%2224665-sample-psis-that-include-risk-assessments%22)
- [GitHub repository with data files and Jupyter notebook for ProPublica analysis of COMPAS risk scores](https://github.com/propublica/compas-analysis)

Discussion questions:
- What data sources did they draw on?
  * Who was involved in collecting this data/information?
  * Why was this data/information originally collected? Or what was the original purpose for data collection?
- Where do we see power at work in this data/information? A lot of directions you could go with this question- how/why the data was collected, categories or classification systems used, etc.
- What data points (or discrete pieces of information) are represented?
- How did the authors organize or filter the data? What transformations happened to get from the original data source to the data structure used in the analysis?
- Other thoughts/comments/questions/observations.

# Data and Environment

We're going to work with two datasets in this lab, both developed and published by the *ProPublica* team.

`compas-scores-two-years.csv`: *ProPublica* selected fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years.

```Python
# loading all the packages
%matplotlib inline

import pandas as pd
import pylab
import numpy as np
import matplotlib.pyplot as plt
```

```Python
# load data
compas_scores_two_years_data = pd.read_csv('compas-scores-two-years.csv', index_col=0)

compas_scores_two_years_data.shape
```

We can see we have 52 columns/fields, and 7,214 rows/records in this dataset.

Compas also offers a score that aims to measure a persons risk of violent recidivism.

We will use the second dataset `compas-scores-two-years-violent` to analyze the violent recidivism score.

```Python
# load data
compas_scores_two_years_violent_data = pd.read_csv('compas-scores-two-years-violent.csv', index_col=0)

compas_scores_two_years_violent_data.shape
```

Again, we can see that we have 52 columns/fields, and 7,214 rows/records in this dataset.

For the first part of the lab, we're actually going to work use RStudio syntax within Python to facilitate easier exploratory data analysis and data wrangling.

We can do this using the `rpy2` package.
- [Click here for more information and documentation on the `rpy2` package](https://pypi.org/project/rpy2/)

First step is to create a new Anaconda environment.

Launch the Anaconda Navigator.

Click on the "Environments" option on the left-hand menu.

Click the "Create" icon to create a new environment.

In the popup, label this new environment and make sure BOTH Python and R are selected. You may need to select `r` from the dropdown menu.

Click the "Create" button to create this environment.

This process may take some time.

Once the new environment with Python and R is ready to go, click on the arrow next to the environment name and select the option to "Open with Jupyter Notebook."

Now we have a Jupyter Notebook environment that can run both Python and RStudio.

For more details on this process: ["Using the R programming language in Jupyter Notebook"](https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/) *Anaconda*



First step is to install the package.

```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install rpy2
```

# Lab Notebook Questions
