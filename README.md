# ProPublica COMPAS Lab

<a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license"><img style="border-width: 0;" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" alt="Creative Commons License" /></a>This tutorial was written by Katherine Walden and is licensed under a <a href="http://creativecommons.org/licenses/by-nc/4.0/" rel="license">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Acknowledgements

This lab is based on the research and technical documentation for *ProPublica*'s 2016 "Machine Bias" article.
- Julia Angwin, Jeff Larson, Surya Mtatu, and Lauren Kirchner, “[Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)” *ProPublica* (23 May 2016). 
- Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin, “[How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)” *ProPublica* (23 May 2016).
- [Link to Google Drive folder that contains the following items:](https://drive.google.com/drive/folders/1-by_FJK2wi86flevOi2WKmomE6wL93yB?usp=sharing)
  * Northpointe document collection gathered by ProPublica team
  * Sentencing reports that include risk assessment, gathered by ProPublica team
- [GitHub repository with data files and Jupyter notebook for ProPublica analysis of COMPAS risk scores](https://github.com/propublica/compas-analysis)

The lab is also adapted from a lab developed by [Lauren F. Klein](https://lklein.com/) for the Spring 2020 Emory University course [QTM 490 "Feminist Data Science"](https://github.com/laurenfklein/feminist-data-science).
- [Lab activity resources (data + Jupyter Notebook)](https://github.com/laurenfklein/feminist-data-science/tree/master/notebooks/lab3-compas)

Klein's lab is based on an exercise by Bucknell University Assistant Professor of Computer Science [Darakhshan Mir](http://eg.bucknell.edu/~djm056/).

# Table of Contents
- [Overview](#overview)
- [Lab Materials](#lab-materials)
- [The Story](#the-story)
- [The Methodology](#the-methodology)
- [Data Sources](#data-sources)
- [Algorithm Audit](#algorithm-audit)
  - [Setup & Environment](#setup--environment)
  - [Data Wrangling](#data-wrangling)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
    - [Unpacking Risk Scores](#unpacking-risk-scores)
    - [Visualizing Risk Scores](#visualizing-risk-scores)
    - [Racial Bias in COMPAS](#racial-bias-in-compas)
    - [Risk of Violent Recidivism](#risk-of-violent-recidivism)
    - [Other Factors](#other-factors)
  - [Predictive Accuracy of COMPAS](#predictive-accuracy-of-compas)
    * [Proportional hazards model](#proportional-hazards-model)
    * [Running the model](#running-the-model)
  - [Directions of the Racial Bias](#directions-of-the-racial-bias)
  - [Gender Differences in COMPAS Scores](#gender-differences-in-compas-scores)
- [Putting it all together](#putting-it-all-together)
- [Critiques of the ProPublica Project](#critiques-of-the-propublica-project)
- [Lab Notebook Components](#lab-notebook-components)

# Overview

In 2016, a *ProPublica* investigative journalism team including published "Machine Bias," an incisive look at the COMPAS risk prediction system used in the criminal justice system. In addition to the main story that emphasizes the human toll of a racially-biased system, the *ProPublica* team published a detailed methodology, data, and technical documentation for the story.

This lab is based on that work.

# Lab Materials

## Procedure
- [Click here](https://colab.research.google.com/drive/1oXj7TJotQ_vPtWNVVSzExo0--QEeLsw0?usp=sharing) to access the lab procedure as a Jupyter Notebook.
- [Click here](https://rstudio.cloud/project/3480951) to access the lab procedure as an RMarkdown file via RStudio Cloud
  * *While the primary lab procedure is written in Python, there is an RMarkdown file that runs through the same workflows from within RStudio*
  * NOTE: If working in RStudio cloud, you'll need to select the `Save a Permanent Copy` icon in the top-right hand corner to make a copy of the project.

## Template
- [Jupyter Notebook](https://colab.research.google.com/drive/1e3ZeCyHOjauNVEW5U0NPq-_exceAKYmy?usp=sharing)
- [Google Doc](https://docs.google.com/document/d/1SzcEiEbTTTyiPDygGP4Bw0OWOwEsN0Y5x7RHfJCP-Vw/copy)

## Data Files

We'll need the `truth_tables.py` file to run some of the calculations later on in this lab.
- [Link to download](https://drive.google.com/file/d/1hH8TfJ1ADcXs7WnrVrzTNPzaoGxeN4qH/view?usp=sharing) from Google Drive
- [Link to download from GitHub](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py)

```Python
# code to download the file within your Python IDE
import json, requests, urllib, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py", "truth_tables.py")
```

We'll also be using four data files.
- `compas-scores-two-years-violent.csv`
  * [Google Drive](https://drive.google.com/file/d/1ONM0NwwCLxeIF0Z23jpyHpBFdQxclGcR/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years-violent.csv)
- `compas-scores-two-years.csv`
  * [Google Drive](https://drive.google.com/file/d/1KgZomaF2Jbob9sW5zrhQs8NcnEd5h2hG/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years.csv)
- `cox-parsed.csv`
  * [Google Drive](https://drive.google.com/file/d/1uGr-5xnRPdcZKHtgCY6qiSDguPjNLzoL/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv)
- `cox-violent-parsed.csv`
  * [Google Drive](https://drive.google.com/file/d/1ewAjZObRNCcx55w6Z4WtbphlIRZdy_X3/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-violent-parsed.csv)

[Link to download all files for this lab from Google Drive as a .zip](https://drive.google.com/file/d/10ntNzhF7c7b-4G1ifeZkM5ThjFeKH-Ja/view?usp=sharing).

# The Story

Read: Julia Angwin, Jeff Larson, Surya Mtatu, and Lauren Kirchner, “[Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)” *ProPublica* (23 May 2016). 

Discussion questions: 
- What stood out to you from the story? 
  * Where and how do we see race and surveillance at work?
- From reading the story, what are you able to tell about how the authors analyzed the technology system?
- How do the authors describe the design and functionality of the technology system?
- Other comments, questions, etc.

# The Methodology

Read: Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin, “[How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)” *ProPublica* (23 May 2016).

Discussion questions:
- What were the goals or central questions for the study?
- Wat data sources did they draw on (and why)?
- How did they approach analyzing the algorithm?
  * Specific questions
  * Statistical methods
  * Visualizations/calculations
- What conclusions do the authors draw from the analysis?
  * Connections we can make with race and surveillance?
- Other comments, questions, observations, etc.

# Data Sources

Explore:
- [Link to Google Drive folder that contains](https://drive.google.com/drive/folders/1-by_FJK2wi86flevOi2WKmomE6wL93yB?usp=sharing)
  * Northpointe document collection gathered by ProPublica team
  * Sentencing reports that include risk assessment, gathered by ProPublica team
- [GitHub repository with data files and Jupyter notebook for ProPublica analysis of COMPAS risk scores](https://github.com/propublica/compas-analysis)

Discussion questions:
- What data sources did the ProPublica team draw on?
  * Who was involved in collecting this data/information?
  * Why was this data/information originally collected? Or what was the original purpose for data collection?
- Where do we see power at work in this data/information? 
  * *A lot of directions you could go with this question- how/why the data was collected, categories or classification systems used, etc.*
- What data points (or discrete pieces of information) are represented?
- How did the authors organize or filter the data? What transformations happened to get from the original data source to the data structure used in the analysis?
- Other comments, questions, observations, etc.

# Algorithm Audit

## Setup & Environment

Throughout this lab, we'll be using RStudio syntax within Python to facilitate easier exploratory data analysis and data wrangling.
- NOTE: If you're working through the lab in RStudio, you'll be using Python syntax from within RStudio.

We can do this using the `rpy2` package.
- [Click here for more information and documentation on the `rpy2` package](https://pypi.org/project/rpy2/)

### If Using Jupyter Notebook/Jupyter Lab (Anaconda)

First step is to create a new Anaconda environment. This process may take some time.
- Launch the Anaconda Navigator.
- Click on the "Environments" option on the left-hand menu.
- Click the "Create" icon to create a new environment.
- In the popup, label this new environment and make sure BOTH Python and R are selected. You may need to select `r` from the dropdown menu.
- Click the "Create" button to create this environment.


Once the new environment with Python and R is ready to go, click on the arrow next to the environment name and select the option to "Open with Jupyter Lab."  Now we have a Jupyter Notebook environment that can run both Python and RStudio.

For more details on this process: ["Using the R programming language in Jupyter Notebook"](https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/) *Anaconda*

### For Everyone

Our next step is to install the `rpy2` package.

```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install rpy2==3.5.1
```

Now we can import the `rpy2` module into Python and set up what is called a `magic command` to run RStudio syntax from within our Python Jupyter Notebook.

```Python
# set up rpy2 magic command
%load_ext rpy2.ipython

# filter rstudio warnings
import warnings
warnings.filterwarnings('ignore')
```

Now, anytime we want to run RStudio code, we can start a code cell with the `%%R` magic command.
- Folks using Anaconda may run into error messages when running the previous block of code. If needed, run the code below and replace the file path with the anaconda path to R from your local computer.

```Python
import os
os.environ['R_HOME'] = '/Users/<your user>/anaconda3/envs/<env name>/lib/R'
```

A couple more R packages to load before we start bringing in the data.

```Python
%%R
# load dplyr package for data wrangling
library(dplyr)

# load ggplot package for visualization
library(ggplot2)
```

"`dplyr` is a grammar of data manipulation, providing a consistent set of verbs that help you solve the most common data manipulation challenges:
  * `mutate()` adds new variables that are functions of existing variables
  * `select()` picks variables based on their names.
  * `filter()` picks cases based on their values.
  * `summarise()` reduces multiple values down to a single summary.
  * `arrange()` changes the ordering of the rows.

These all combine naturally with `group_by()` which allows you to perform any operation “by group”. You can learn more about them in [`vignette("dplyr")`](https://dplyr.tidyverse.org/articles/dplyr.html). As well as these single-table verbs, dplyr also provides a variety of two-table verbs, which you can learn about in [`vignette("two-table")`](https://dplyr.tidyverse.org/articles/two-table.html).” [Source: [dplyr.tidyverse.org](https://dplyr.tidyverse.org/)]
- More dplyr documentation: [cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html](https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html)
- For more on the conceptual foundations for data transformation in `R`: [Chapter 5, "Data Transformation"](https://r4ds.had.co.nz/transform.html) in Hadley Wickham and Garrett Grolemund, [*R for Data Science: Visualize, Model, Transform, Tidy, and Import Data*](https://r4ds.had.co.nz/index.html) (O'Reilly, 2017).

“R has several systems for making graphs, but `ggplot2` is one of the most elegant and most versatile. ggplot2 implements the grammar of graphics, a coherent system for describing and building graphs. With ggplot2, you can do more faster by learning one system and applying it in many places.” [[Chapter 3 “Data Visualization”](https://r4ds.had.co.nz/data-visualisation.html) in Garrett Grolemund and Hadley Wickham, *R for Data Science*]

“`ggplot2` is a system for declaratively creating graphics, based on The Grammar of Graphics. You provide the data, tell ggplot2 how to map variables to aesthetics, what graphical primitives to use, and it takes care of the details...It’s hard to succinctly describe how ggplot2 works because it embodies a deep philosophy of visualisation. However, in most cases you start with ggplot(), supply a dataset and aesthetic mapping (with aes()). You then add on layers (like geom_point() or geom_histogram()), scales (like scale_colour_brewer()), faceting specifications (like facet_wrap()) and coordinate systems (like coord_flip()).” [[ggplot2.tidyverse.org](https://ggplot2.tidyverse.org/)]

We're also going to install and load a few Python packages to have on hand for later in the lab.

```Python
# install matplotlib
import sys
!{sys.executable} -m pip install matplotlib
```

```Python
# load all the things!
%matplotlib inline

import pandas as pd
import pylab
import numpy as np
import matplotlib.pyplot as plt
```

## Data Wrangling

### Loading

We're going to work with two datasets in this lab, both developed and published by the *ProPublica* team. In `compas-scores-two-years.csv`, the *ProPublica* team selected fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years. We can see we have 52 columns/fields, and 7,214 rows/records in this dataset.

**RStudio Syntax**

```Python
%%R

# load data from CSV file using R dataframe syntax
# compas_two_year_scores <- read.csv("compas-scores-two-years.csv")

# load data from URL using R dataframe syntax
compas_two_year_scores <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years.csv")

# show data dimensions
nrow(compas_two_year_scores)
```

**Python Syntax**

We can also express these steps programatically in Python using `pandas`:

```Python
# load data from file
# compas_two_year_scores = pd.read_csv('compas-scores-two-years.csv', index_col=0)

# load data from url
compas_two_year_scores = pd.read_csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years.csv", index_col = 0)

# show data shape
compas_two_year_scores.shape
```

```Python
# look at the first five rows of the dataset
pd.options.display.max_columns = None # have to do this otherwise it limits the number of cols shown

# show first five rows
compas_two_year_scores.head()
```

### Filtering

Not all of the rows are useable for the first round of analysis. The *ProPublica* team determined a number of criteria for removing missing or unusable data. 

These criteria are listed below:
- If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
- We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
- In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
- We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

To filter the data using these criteria...

**RStudio Syntax**

```Python
%%R
# create new dataframe using filtering criteria
df <- dplyr::select(compas_two_year_scores, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% 
        filter(days_b_screening_arrest <= 30) %>%
        filter(days_b_screening_arrest >= -30) %>%
        filter(is_recid != -1) %>%
        filter(c_charge_degree != "O") %>%
        filter(score_text != 'N/A')

# show updated dataframe
nrow(df)
```

**Python Pandas Syntax**

To run that filtering operation in Python, first we'll filter out those which do not have a COMPAS-scored case, as indicated by the recidivist flag `is_recid` set at -1.

```Python
# filter data
filterData = compas_two_year_scores[(compas_two_year_scores['is_recid'] != -1)]

# show updated data shape
filterData.shape
```

### More Filtering

Within the cases with a COMPAS score, we also need to check to see if we have the right offense. So if the charge date of a defendant's COMPAS-scored crime was not within 30 days from when the person was arrested, it's best to assume that we do not have the right offense, and remove that row. We will filter out rows where **days_b_screening_arrest** is over 30 or under -30:

**Python Pandas Syntax**

```Python
# filter data
filterData = compas_two_year_scores[(compas_two_year_scores['days_b_screening_arrest'] <= 30) & (compas_two_year_scores['days_b_screening_arrest'] >= -30)]

# show updated data
filterData.shape
```

The results of both filtering processes should be the same: 52 columns or fields, and 6172 observations or rows/records.

## Exploratory Data Analysis

### Unpacking Risk Scores

Higher COMPAS scores are slightly correlated with a longer length of stay. 

**RStudio Syntax**

```Python
%%R
# select length of stay
df$length_of_stay <- as.numeric(as.Date(df$c_jail_out) - as.Date(df$c_jail_in))

# show correlation
cor(df$length_of_stay, df$decile_score)
```

After filtering we have the following demographic breakdown:

#### By Age

**RStudio Syntax**

```Python
%%R
# summary of age range represented in the dataset
summary(df$age_cat)
```

**Python Pandas Syntax**

```Python
# Python syntax for age summary
filterData.age_cat.value_counts()
```

#### By Race

**Rstudio Syntax**

```Python
%%R
# summary of race/ethnicity information represented in the dataset
summary(df$race)
```

**Python Pandas Syntax**

```Python
# race/ethnicity summary
filterData.race.value_counts()
```

```Python
# show percentages by race/ethnicity
print("Black defendants: %.2f%%" %            (3175 / 6172 * 100))
print("White defendants: %.2f%%" %            (2103 / 6172 * 100))
print("Hispanic defendants: %.2f%%" %         (509  / 6172 * 100))
print("Asian defendants: %.2f%%" %            (31   / 6172 * 100))
print("Native American defendants: %.2f%%" %  (11   / 6172 * 100))
```

#### By Risk Score

**RStudio Syntax**
```Python
%%R
# summary of risk scores
summary(df$score_text)
```

```Python
%%R
# cross tab summary of data by race/ethnicity and gender
xtabs(~ sex + race, data=df)
```

**Python Pandas Syntax**
```Python
# risk score summary
filterData.score_text.value_counts()
```

```Python
# cross tabs for recidivsm rates by race
pd.crosstab(filterData.sex, filterData.race)
```

#### By Gender

**RStudio Syntax**
```Python
%%R
# summary of data by gender
summary(df$sex)
```

**Python Pandas Syntax**
```Python
# gender breakdown
filterData.sex.value_counts()
```

```Python
# percentages by gender
print("Men: %.2f%%" %   (4997 / 6172 * 100))
print("Women: %.2f%%" % (1175 / 6172 * 100))
```

#### By Recidivism Rate

**RStudio Syntax**
```Python
%%R
# number of rows where two_year_recid = 1
nrow(filter(df, two_year_recid == 1))
```

```Python
%%R
# percentage of rows where two_year_recid = 2
nrow(filter(df, two_year_recid == 1)) / nrow(df) * 100
```

### Visualizing Risk Scores

Judges are often presented with two sets of scores from the Compas system -- one that classifies people into High, Medium and Low risk, and a corresponding decile score. *ProPublica*'s analysis found a clear downward trend in the decile scores as those scores increase for white defendants.

**Calculating decile risk scores by race, using RStudio syntax**

```Python
%%R
# create bar chart with decile risk scores for Black defendants
pblack <- ggplot(data=filter(df, race =="African-American"), aes(ordered(decile_score))) + 
          geom_bar() + xlab("Decile Score") +
          ylim(0, 650) + ggtitle("Black Defendant's Decile Scores")

# create bar chart with decile risk scores for white defendants
pwhite <- ggplot(data=filter(df, race =="Caucasian"), aes(ordered(decile_score))) + 
          geom_bar() + xlab("Decile Score") +
          ylim(0, 650) + ggtitle("White Defendant's Decile Scores")

# show first bar chart
show(pblack)

# show second bar chart
show(pwhite)
```

```Python
%%R
# cross tab data by race and decile score
xtabs(~ decile_score + race, data=df)
```

**Calculating decile risk scores by race, using Python Pandas syntax**

```Python
# cross tabs by race/ethnicity
scores_by_race = pd.crosstab(filterData.race, filterData.decile_score)

# show scores
scores_by_race
```

```Python
# another visualization with the decile scores disaggregated by race/ethnicity

# create labels
labels = list(scores_by_race.columns)

# list with scores by race
aa_scores = list(scores_by_race.loc["African-American"])
c_scores = list(scores_by_race.loc["Caucasian"])

# arrange labels and set width
x = np.arange(len(labels))
width = 0.35

# generate plot
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, aa_scores, width, label='African-American')
rects2 = ax.bar(x + width/2, c_scores, width, label='Caucasian')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Scores by decile and race')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# function for placing labels
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# run label placing functions
autolabel(rects1)
autolabel(rects2)

# update figure layout
fig.tight_layout()

# show figure
plt.show()
```

### Racial Bias in COMPAS

These visualizations suggest that *something* is going on. But in order to test our intution that there is a significant difference in COMPAS scores across different racial categories, we need to run a logistic regression, comparing low scores to high scores.

After filtering out unusable rows, *ProPublica*'s next step was whether there is a significant difference in Compas scores between races. They explored this question by changing some variables into factors, and running a logistic regression, comparing low scores to high scores.

These factor conversions were necessary because of RStudio syntax. The first step would be to convert the `c_charge_degree`, `age_cat`, `race`, `sex` (which are all categorical data) into factors. 

**RStudio Syntax**

*NOTE from Prof. Walden- I'm getting a `RInterpreter Error` for the code block below related to `LAPACK routines`. If you're getting, jump to the `Python Statsmodels syntax` code cells.*

```Python
%%R
# filter dataframe
df <- mutate(df, crime_factor = factor(c_charge_degree)) %>%
      mutate(age_factor = as.factor(age_cat)) %>%
      within(age_factor <- relevel(age_factor, ref = 1)) %>%
      mutate(race_factor = factor(race)) %>%
      within(race_factor <- relevel(race_factor, ref = 3)) %>%
      mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%
      within(gender_factor <- relevel(gender_factor, ref = 2)) %>%
      mutate(score_factor = factor(score_text != "Low", labels = c("LowScore","HighScore")))
model <- glm(score_factor ~ gender_factor + age_factor + race_factor +
                            priors_count + crime_factor + two_year_recid, family="binomial", data=df)

# show updated dataframe
summary(model)
```

But we can use the Patsy API, part of the Python `statsmodels` library, to embed these transformations within the forumla.

**Python Statsmodels Syntax**

```Python
# install statsmodels library
import sys
!{sys.executable} -m pip install statsmodels
```

```Python
# load all the things
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
```

```Python
# create a score variable where "Low" => 0, "Medium"/"High" => 1
filterData['score'] = (filterData['score_text'] != "Low") * 1
```

```Python
# use the Patsy API for formula generation
formula = "score ~ C(sex, Treatment('Male')) + age_cat + " + \
          "C(race, Treatment('Caucasian')) + priors_count + " + \
          "c_charge_degree + two_year_recid"
```

```Python
# run the model
model = smf.glm(formula=formula, data=filterData, family=sm.families.Binomial()).fit()

# show model summary
model.summary()
```

This analysis found Black defendants are 45% more likely than white defendants to receive a higher score correcting for the seriousness of their crime, previous arrests, and future criminal behavior.

```Python
%%R
# control calculations by race
control <- exp(-1.52554) / (1 + exp(-1.52554))
exp(0.47721) / (1 - control + (control * exp(0.47721)))
```

Women are 19.4% more likely than men to get a higher score.

```Python
%%R
# control calculations by gender
exp(0.22127) / (1 - control + (control * exp(0.22127)))
```

Most surprisingly, people under 25 are 2.5 times as likely to get a higher score as middle aged defendants.

```Python
%%R
# control calculations by age
exp(1.30839) / (1 - control + (control * exp(1.30839)))
```

### Risk of Violent Recidivism

COMPAS also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score. The *ProPublica* team used a logistic regression with the `compas-scores-two-years-violent` dataset to test for racial bias in the violent recidivism score.

**RStudio Syntax**

```Python
%%R
# load data from CSV file
# compas_two_year_scores_violent <- read.csv("compas-scores-two-years-violent.csv")

# load data from URL
compas_two_year_scores_violent <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years-violent.csv")

# show data dimensions
nrow(compas_two_year_scores_violent)
```

**Python Pandas Syntax**

```Python
# load data from file
# compas_two_year_scores_violent = pd.read_csv('compas-scores-two-years-violent.csv', index_col=0)

# load data from url
compas_two_year_scores_violent = pd.read_csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years-violent.csv", index_col = 0)

# show data shape
compas_two_year_scores_violent.shape
```

```Python
# look at the first five rows of the dataset
pd.options.display.max_columns = None # have to do this otherwise it limits the number of cols shown

# show first five rows
compas_two_year_scores_violent.head()
```

Again, we can see that we have 52 columns/fields, and 7,214 rows/records in this dataset. As before, the *ProPublica* team determined a number of criteria for removing missing or unusable data. 

These criteria are listed below:
- If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
- We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
- In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
- We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

To filter the data using these criteria:

**RStudio Syntax**

```Python
%%R
# filter dataframe
df1 <- dplyr::select(compas_two_year_scores_violent, age, c_charge_degree, race, age_cat, v_score_text, sex, priors_count, 
                    days_b_screening_arrest, v_decile_score, is_recid, two_year_recid) %>% 
        filter(days_b_screening_arrest <= 30) %>%
        filter(days_b_screening_arrest >= -30) %>% 
        filter(is_recid != -1) %>%
        filter(c_charge_degree != "O") %>%
        filter(v_score_text != 'N/A')

# show updated dataframe
nrow(df1)
```

**Python Pandas Syntax**

To be able to run a similar program in Python, we need to load the `truth_tables.py` file with named functions created by the *ProPublica* team.
- [Link to download](https://drive.google.com/file/d/1hH8TfJ1ADcXs7WnrVrzTNPzaoGxeN4qH/view?usp=sharing) from Google Drive
- [Link to download from GitHub](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py)

If working with Jupyter Notebooks on your local computer, you'll need to move the `truth_tables.py` file into the same directory (folder) as the Jupyter Notebook.
- Alternatively, you can provide the full file path.

If working in Google CoLab, you'll either need to upload the file to your session or mount Google Drive to access the file.
- [Uploading files](https://youtu.be/6HFlwqK3oeo?t=177)
- [Mounting Google Drive](https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/)

Alternatively, you can run the code below to download the file to your working directory.

```Python
# code to download the file within your Python IDE
import json, requests, urllib, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py", "truth_tables.py")
```

We'll also need to have the `cox-violent-parsed.csv` file in your local working directory. The steps above also work, but the code below will download the file programmatically.
- File download options
  * [Google Drive](https://drive.google.com/file/d/1ewAjZObRNCcx55w6Z4WtbphlIRZdy_X3/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-violent-parsed.csv)

```Python
# code to download the cox-violent-parsed.csv within your Python IDE
import json, requests, urllib, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-violent-parsed.csv", "cox-violent-parsed.csv")
```

Then, we can import functions from this file using `from truth_tables import...`.

```Python
# import functions from truth tables
from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable

# import CSV module
from csv import DictReader

# create empty dictionary
vpeople = []

# load data
with open("cox-violent-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                vpeople.append(p)
    except StopIteration:
        pass

# filter for specific condtiions
vpop = list(filter(lambda i: ((i.violent_recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.vscore_valid, vpeople))))

# filter for specific conditions
vrecid = list(filter(lambda i: i.violent_recidivist == True and i.lifetime <= 730, vpeople))

# create dataset with filtered results
vrset = set(vrecid)
vsurv = [i for i in vpop if i not in vrset]
```

```Python
# show updated data
print("All defendants")
vtable(list(vrecid), list(vsurv))
```

The *ProPublica* team found these trends were further exacerbated for Black defendants.

```Python
# show table with violent risk scores for Black defendants
print("Black defendants")
is_afam = is_race("African-American")
vtable(list(filter(is_afam, vrecid)), list(filter(is_afam, vsurv)))
```

```Python
# show table with violent risk scores for white defendants
print("White defendants")
is_white = is_race("Caucasian")
vtable(list(filter(is_white, vrecid)), list(filter(is_white, vsurv)))
```

The *ProPublica* team found that Black defendants were twice as likely to be false positives for a Higher violent score than white defendants.

```Python
38.14 / 18.46
```

They also found white defendants were 63% more likely to get a lower score and commit another crime than Black defendants.

```Python
62.62 / 38.37
```

### Other Factors

```Python
%%R
# age distribution
summary(df1$age_cat)
```

```Python
%%R
# race/ethnicity distribution
summary(df1$race)
```

```Python
%%R
# number of rows where two_year_recid = 1
nrow(filter(df1, two_year_recid == 1))
```

```Python
%%R
# percentage of rows where two_year_recid = 1
nrow(filter(df1, two_year_recid == 1)) / nrow(df) * 100
```

*ProPublica*'s analysis found a clear downward trend in the violent decile scores as those scores increase for white defendants.

```Python
%%R
# create bar chart with decile violent risk scores for Black defendants
pblack_v <- ggplot(data=filter(df1, race =="African-American"), aes(ordered(v_decile_score))) + 
          geom_bar() + xlab("Violent Decile Score") +
          ylim(0, 650) + ggtitle("Black Defendant's Violent Decile Scores")

# create bar chart with decile risk scores for white defendants
pwhite_v <- ggplot(data=filter(df1, race =="Caucasian"), aes(ordered(v_decile_score))) + 
          geom_bar() + xlab("Violent Decile Score") +
          ylim(0, 650) + ggtitle("White Defendant's Violent Decile Scores")

# show first bar chart
show(pblack_v)

# show second bar chart
show(pwhite_v)
```

Again, the *ProPublica* team explored the question of racial bias by changing some variables into factors, and running a logistic regression, comparing low scores to high scores.

**RStudio Syntax**

*NOTE from Prof. Walden- I'm getting a `RInterpreter Error` for the code block below related to `LAPACK routines`. If you're getting, skip this code cell.*

```Python
%%R
# filter data
df1 <- mutate(df1, crime_factor = factor(c_charge_degree)) %>%
      mutate(age_factor = as.factor(age_cat)) %>%
      within(age_factor <- relevel(age_factor, ref = 1)) %>%
      mutate(race_factor = factor(race,
                                  labels = c("African-American", 
                                             "Asian",
                                             "Caucasian", 
                                             "Hispanic", 
                                             "Native American",
                                             "Other"))) %>%
      within(race_factor <- relevel(race_factor, ref = 3)) %>%
      mutate(gender_factor = factor(sex, labels= c("Female","Male"))) %>%
      within(gender_factor <- relevel(gender_factor, ref = 2)) %>%
      mutate(score_factor = factor(v_score_text != "Low", labels = c("LowScore","HighScore")))
model <- glm(score_factor ~ gender_factor + age_factor + race_factor +
                            priors_count + crime_factor + two_year_recid, family="binomial", data=df)

# show model summary
summary(model)
```

The violent score overpredicts recidivism for Black defendants by 77.3% compared to white defendants.

```Python
%%R
# control calculations by race
control <- exp(-2.24274) / (1 + exp(-2.24274))
exp(0.65893) / (1 - control + (control * exp(0.65893)))
```

Defendands under 25 are 7.4 times as likely to get a higher score as middle aged defendants.

```Python
%%R
# control calculations by age
exp(3.14591) / (1 - control + (control * exp(3.14591)))
```

## Predictive Accuracy of COMPAS

To test whether Compas scores do an accurate job of deciding whether an offender is Low, Medium or High risk,  *ProPublica* ran a Cox Proportional Hazards model. Northpointe, the company that created COMPAS and markets it to Law Enforcement, also ran a Cox model in their [validation study](http://cjb.sagepub.com/content/36/1/21.abstract).

*ProPublica* used the counting model and removed people when they were incarcerated. Due to errors in the underlying jail data, they ended up filtering out 32 rows that have an end date more than the start date. They determined that since there are 13,334 total rows in the data, such a small amount of errors would not affect the results.

### Proportional Hazards Model

What is a Cox Proportional Hazards model?
- "**Survival analysis** is a branch of statistics for analyzing the expected duration of time until one or more events happen, such as death in biological organisms and failure in mechanical systems. This topic is called reliability theory or reliability analysis in engineering, duration analysis or duration modelling in economics, and event history analysis in sociology. Survival analysis attempts to answer certain questions, such as what is the proportion of a population which will survive past a certain time? Of those that survive, at what rate will they die or fail? Can multiple causes of death or failure be taken into account? How do particular circumstances or characteristics increase or decrease the probability of survival?" ([Wikipedia](https://en.wikipedia.org/wiki/Survival_analysis))

- "**Proportional hazards models** are a class of survival models in statistics. Survival models relate the time that passes, before some event occurs, to one or more covariates that may be associated with that quantity of time. In a proportional hazards model, the unique effect of a unit increase in a covariate is multiplicative with respect to the hazard rate. For example, taking a drug may halve one's hazard rate for a stroke occurring, or, changing the material from which a manufactured component is constructed may double its hazard rate for failure" ([Wikipedia](https://en.wikipedia.org/wiki/Proportional_hazards_model)).

The Cox Proportional Hazards model was developed by British statistician Sir David Cox in the 1970s. 
- For more background on the model: D.R. Cox, "[Regression Models and Life Tables]( http://www.jstor.org.proxy.library.nd.edu?url=https://www.jstor.org/stable/2985181)" *Journal of the Royal Statistical Society* 34:2 (1972): 187-220.

The model is as regression model most often used to determine the association or relationship between patient survival time and predictor variables.
- [For more information on the underlying math in the Cox model](https://en.wikipedia.org/wiki/Proportional_hazards_model#The_Cox_model)
- [Examples of the Cox model used to analyze health outcomes](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Survival/BS704_Survival6.html)

### Running the Model

To run this model, we need a couple of additional R packages.
- The `ggfortify` package takes a curve and points object and converts it to a data frame that can be plotted using `ggplot2`. [For more on `ggfortify`](https://cran.r-project.org/web/packages/ggfortify/index.html)

- The `survival` package contains the definition for the Cox model (as well as other statistical models). [For more on `survival`](https://cran.r-project.org/web/packages/survival/index.html)

**Terminal Syntax to Install RStudio Packages in Jupyter Notebook**

```
! R -e "install.packages('ggfortify')"
! R -e "install.packages('survival')"
! R -e "library(ggfortify)"
! R -e "library(survival)"
```

**RStudio Syntax**

```Python
%%R
# load packages
library(survival)
library(ggfortify)
```

We also need to load data structured for the model. We can see this data has 13,419 observations.

**RStudio Syntax**

```Python
%%R
# load data from CSV file
# cox_parsed <- read.csv("cox-parsed.csv")

# load data from URL
cox_parsed <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv")

# show data dimensions
nrow(cox_parsed)
```

**Python Pandas Syntax**

```Python
# load data from file
# cox_parsed = pd.read_csv('cox-parsed.csv', index_col=0)

# load data from url
cox_parsed = pd.read_csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv", index_col = 0)

# show data shape
cox_parsed.shape
```

```Python
# look at the first five rows of the dataset
pd.options.display.max_columns = None # have to do this otherwise it limits the number of cols shown

# show first five rows
cox_parsed.head()
```

The next step taken by the ProPublica team was filtering the data 

```Python
%%R
# filter data
cox_parsed <- filter(filter(read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv"), score_text != "N/A"), end > start) %>%
        mutate(race_factor = factor(race,
                                  labels = c("African-American", 
                                             "Asian",
                                             "Caucasian", 
                                             "Hispanic", 
                                             "Native American",
                                             "Other"))) %>%
        within(race_factor <- relevel(race_factor, ref = 3)) %>%
        mutate(score_factor = factor(score_text)) %>%
        within(score_factor <- relevel(score_factor, ref=2))

# remove duplicate rows
grp <- cox_parsed[!duplicated(cox_parsed$id),]

# show updated data
nrow(grp)
```

The results of that filtering is 10,314 observations.

```Python
%%R
# score ranges
summary(grp$score_factor)
```

```Python
%%R
# data summary by race/ethnicity
summary(grp$race_factor)
```

```Python
%%R
# score factor
f <- Surv(start, end, event, type="counting") ~ score_factor

# create model
model <- coxph(f, data=cox_parsed)

# model summary
summary(model)
```

People placed in the `High` category are 3.5 times as likely to recidivate, and the COMPAS system's concordance 63.6%. This is lower than the accuracy quoted in the Northpoint study of 68%.

```Python
%%R
# decile scores
decile_f <- Surv(start, end, event, type="counting") ~ decile_score

# create decile score model
dmodel <- coxph(decile_f, data=cox_parsed)

# show decile score model
summary(dmodel)
```

COMPAS's decile scores are a bit more accurate at 66%.

*ProPublica* tested if the algorithm behaved differently across races by including a race interaction term in the cox model.

**RStudio Syntax**

```Python
%%R
# score factor
f2 <- Surv(start, end, event, type="counting") ~ race_factor + score_factor + race_factor * score_factor

# create model
model <- coxph(f2, data=cox_parsed)

# show model summary
print(summary(model))
```

The *ProPublica* team found that the interaction term shows a similar disparity as the logistic regression above. They also found high risk white defendants are 3.61 more likely than low risk white defendants, while High risk black defendants are 2.99 more likely than low.

**Python Math Syntax**

```Python
# import statement
import math

# show interaction term by race
print("Black High Hazard: %.2f" % (math.exp(-0.18976 + 1.28350)))
print("White High Hazard: %.2f" % (math.exp(1.28350)))
print("Black Medium Hazard: %.2f" % (math.exp(0.84286-0.17261)))
print("White Medium Hazard: %.2f" % (math.exp(0.84286)))
```

**RStudio Syntax**

```Python
%%R -w 900 -h 563 -u px
# create fitted curve based on survival model
fit <- survfit(f, data=cox_parsed)

# plot curve
plotty <- function(fit, title) {
  return(autoplot(fit, conf.int=T, censor=F) + ggtitle(title) + ylim(0,1))
}

# show plot
plotty(fit, "Overall")
```

The *ProPublica* team found that Black defendants do recidivate at higher rates according to race specific Kaplan Meier plots.

**RStudio Syntax**

```Python
%%R
# filter data for white defendants
white <- filter(cox_parsed, race == "Caucasian")
white_fit <- survfit(f, data=white)

# filter data for Black defendants
black <- filter(cox_parsed, race == "African-American")
black_fit <- survfit(f, data=black)
```

```Python
%%R
# plot white defendants
plotty(white_fit, "White defendants")
```

```Python
%%R
# plot Black defendants
plotty(black_fit, "Black defendants")
```

```Python
%%R
# calculate model summary
summary(fit, times=c(730))
```

```Python
%%R
# calculate model summary for Black defendants
summary(black_fit, times=c(730))
```

```Python
%%R
# calculate model summary for white defendnats
summary(white_fit, times=c(730))
```

The *ProPublica* team found that race specific models had similar concordance values.

**RStudio Syntax**

```Python
%%R
# calculate concordance values for white defendnats
summary(coxph(f, data=white))
```

```Python
%%R
# calculate concordance values for Black defendants
summary(coxph(f, data=black))
```

The *ProPublica* team found that Compas's violent recidivism score has a slightly higher overall concordance score of 65.1%.

**RStudio Syntax**

```Python
%%R
# load and filter data
violent_data <- filter(filter(read.csv("cox-violent-parsed.csv"), score_text != "N/A"), end > start) %>%
        mutate(race_factor = factor(race,
                                  labels = c("African-American", 
                                             "Asian",
                                             "Caucasian", 
                                             "Hispanic", 
                                             "Native American",
                                             "Other"))) %>%
        within(race_factor <- relevel(race_factor, ref = 3)) %>%
        mutate(score_factor = factor(score_text)) %>%
        within(score_factor <- relevel(score_factor, ref=2))


# create survival model
vf <- Surv(start, end, event, type="counting") ~ score_factor

# create concordance values
vmodel <- coxph(vf, data=violent_data)

# remove duplicates
vgrp <- violent_data[!duplicated(violent_data$id),]

# print output
print(nrow(vgrp))

# model summary
summary(vmodel)
```

The *ProPublica* team found that in this case, there isn't a significant coefficient on Black/African American defendants with High Scores.

**RStudio Syntax**

```Python
%%R
# survival model
vf2 <- Surv(start, end, event, type="counting") ~ race_factor + race_factor * score_factor

# concordance values
vmodel <- coxph(vf2, data=violent_data)

# model summary 
summary(vmodel)
```

```Python
%%R
# concordance values summary for Black defendants
summary(coxph(vf, data=filter(violent_data, race == "African-American")))
```

```Python
%%R
# concordance values summary for white defendants
summary(coxph(vf, data=filter(violent_data, race == "Caucasian")))
```

```Python
%%R
# filter data for white defendnats
white <- filter(violent_data, race == "Caucasian")

# create survival model for white defendants
white_fit <- survfit(vf, data=white)

# filter data for black defendants
black <- filter(violent_data, race == "African-American")

# create survival model for black defendants
black_fit <- survfit(vf, data=black)
```

```Python
%%R
# plot white defendants
plotty(white_fit, "White defendants")
```

```Python
%%R
# plot Black defendants
plotty(black_fit, "Black defendants")
```

## Directions of the Racial Bias

*ProPublica*'s analysis found that the COMPAS algorithm does overpredict Black defendants' future recidivism. The next section of the lab looks at how they explored the direction of the bias. Their analysis found fine differences in overprediction and underprediction by comparing COMPAS scores across racial lines.

To be able to run a similar program in Python, we need to load the `truth_tables.py` file with named functions created by the *ProPublica* team.
- [Link to download](https://drive.google.com/file/d/1hH8TfJ1ADcXs7WnrVrzTNPzaoGxeN4qH/view?usp=sharing) from Google Drive
- [Link to download from GitHub](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py)

If working with Jupyter Notebooks on your local computer, you'll need to move the `truth_tables.py` file into the same directory (folder) as the Jupyter Notebook.
- Alternatively, you can provide the full file path.

If working in Google CoLab, you'll either need to upload the file to your session or mount Google Drive to access the file.
- [Uploading files](https://youtu.be/6HFlwqK3oeo?t=177)
- [Mounting Google Drive](https://www.marktechpost.com/2019/06/07/how-to-connect-google-colab-with-google-drive/)

Alternatively, you can run the code below to download the file to your working directory.

```Python
# code to download the file within your Python IDE
import json, requests, urllib, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/truth_tables.py", "truth_tables.py")
```

We'll also need to have the `cox-violent-parsed.csv` file in your local working directory. The steps above also work, but the code below will download the file programmatically.
- File download options
  * [Google Drive](https://drive.google.com/file/d/1uGr-5xnRPdcZKHtgCY6qiSDguPjNLzoL/view?usp=sharing)
  * [GitHub URL](https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv)

```Python
# code to download the cox-parsed.csv within your Python IDE
import json, requests, urllib, urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv", "cox-parsed.csv")
```

Then, we can import functions from this file using `from truth_tables import...`.

**Python Syntax**

```Python
# import functions from truth tables
from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable

# import CSV module
from csv import DictReader

# create empty dictionary
people = []

# load parsed data
with open("cox-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                people.append(p)
    except StopIteration:
        pass

# filter for specific conditions
pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.score_valid, people))))

# filter for specific fonditions
recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop))

rset = set(recid)

# show survival score
surv = [i for i in pop if i not in rset]
```

```Python
# print table risk scores
print("All defendants")
table(list(recid), list(surv))
```

```Python
# print percentage of total population
print("Total pop: %i" % (2681 + 1282 + 1216 + 2035))
```

```Python
# import statistics module
import statistics

# print followup time
print("Average followup time %.2f (sd %.2f)" % (statistics.mean(map(lambda i: i.lifetime, pop)),
                                                statistics.stdev(map(lambda i: i.lifetime, pop))))

# print median followup time
print("Median followup time %i" % (statistics.median(map(lambda i: i.lifetime, pop))))
```

Overall, the false positive rate is 32.35%.

```Python
# create table with risk scores for Black defendants
print("Black defendants")
is_afam = is_race("African-American")
table(list(filter(is_afam, recid)), list(filter(is_afam, surv)))
```

That number is higher for Black defendants at 44.85%.

```Python
# create table with risk scores for white defendants
print("White defendants")
is_white = is_race("Caucasian")
table(list(filter(is_white, recid)), list(filter(is_white, surv)))
```

And lower for white defendants at 23.45%.

```Python
44.85 / 23.45
```

In the *ProPublica* team's analysis, these results mean under COMPAS Black defendants are 91% more likely to get a higher score and not go on to commit more crimes than white defendants after two years.

They also found that COMPAS scores misclassify white reoffenders as low risk at 70.4% more often than Black reoffenders.

**Python Syntax**

```Python
47.72 / 27.99
```

```Python
# create table for white defendants' risk scores
hightable(list(filter(is_white, recid)), list(filter(is_white, surv)))
```

```Python
# create table for Black defendants' risk scores
hightable(list(filter(is_afam, recid)), list(filter(is_afam, surv)))
```

## Gender Differences in COMPAS Scores

The *ProPublica* team used gender-specific Kaplan Meier estimates to look at differences between men and women in terms of underlying recidivism rates.

**RStudio Syntax**

```Python
%%R
# filter by gender
female <- filter(cox_parsed, sex == "Female")
male   <- filter(cox_parsed, sex == "Male")
male_fit <- survfit(f, data=male)
female_fit <- survfit(f, data=female)
```

```Python
%%R
# show summary for male defendants
summary(male_fit, times=c(730))
```

```Python
%%R
# show summary for female defendants
summary(female_fit, times=c(730))
```

```Python
%%R 
# plot female defendants
plotty(female_fit, "Female")
```

```Python
%%R 
# plot male defendants
plotty(male_fit, "Male")
```

From these plots, the *ProPublica* team determined the Compas score treats a `High risk woman` the same as a `Medium risk man`.

# Putting It All Together

Let's take a step back here and think about what these results mean, or at least what they mean in relation to the conclusions drawn in the *ProPublica* article.

Discussion questions:
- What conclusions would you draw about the COMPAS algorithm based on this analysis?
- What other types of questions would you want to ask or what else do you want to know about the algorithm?
- How does the analysis outlined in this lab and the ProPublica methodology white paper relate to the arguments/conclusions presented in the "Machine Bias" article?
- What limitations or shortcomings would you identify for this analysis (the analysis itself, the statistical models used, the input data, etc.)?
- Other comments, questions, observations, etc.

# Critiques of the ProPublica Project

There have been some critiques and responses to *ProPublica*'s methodology, specifically around the data analysis component.

If you're interested in exploring these critiques:
- Northpointe:
  * Northpointe Suite, "[Response to ProPublica: Demonstrating Accuracy Equity, and Predictive Parity](https://www.equivant.com/response-to-propublica-demonstrating-accuracy-equity-and-predictive-parity/)" *Equivant blog* (8 July 2016)
  * Northpointe Inc. Research Department, "[Response to ProPublica: Demonstrating Accuracy Equity, and Predictive Parity](http://go.volarisgroup.com/rs/430-MBX-989/images/ProPublica_Commentary_Final_070616.pdf/)" *Research Paper* (8 July 2016)

- Llewellyn Hinkes Jones, "[ProPublica's Misleading Machine Bias](https://medium.com/@llewhinkes/propublicas-misleading-machine-bias-19c971549a18)" *Medium blog* (6 October 2020)

- Matias Barenstein (Economist at Federal Trade Commission)
  * [GitHub repository](https://github.com/mbarenstein/ProPublica_COMPAS_Data_Revisited)
  * [preprint research article deposited with arXiv](https://arxiv.org/abs/1906.04711)
  * "[The Data Processing Error in a Prominent Fair Machine Learning Dataset](https://towardsdatascience.com/the-data-processing-error-in-the-most-prominent-fair-machine-learning-dataset-short-version-d27d8d390fea)" *Towards Data Science* (22 August 2019)

Discussion questions:
- What are some of your initial observations or thoughts on these critiques?
- How do these critiques respond to or engage with the ProPublica investigation?
- What are some of your takeaways from exploring these critiques?
- Particularly, how are you thinking about what it looks like to analyze, investigate, or hold accountable “black box” tech systems?
- Other comments, questions, observations, etc.

# Lab Notebook Components

Lab Notebook Template:
- [Jupyter Notebook](https://colab.research.google.com/drive/1e3ZeCyHOjauNVEW5U0NPq-_exceAKYmy?usp=sharing)
- [Google Doc](https://docs.google.com/document/d/1SzcEiEbTTTyiPDygGP4Bw0OWOwEsN0Y5x7RHfJCP-Vw/copy)

The lab notebook consists of a narrative that documents and describes your experience working through this lab.

You can respond to/engage with the discussion questions embedded throughout the lab procedure.

Other questions for the lab notebook: 
- What challenges did you face, and how did you solve them?
- What did you learn about machine learning/predictive models through this lab?
- How are you thinking about the *ProPublica* article and investigation after this lab?
- How are you thinking about race and surveillance after this case study/lab?
- Other comments/questions/observations

I encourage folks to include code and/or screenshots as part of that narrative.
- You are welcome (but not required) to include code as part of that narrative.
