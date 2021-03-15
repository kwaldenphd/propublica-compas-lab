# ProPublica COMPAS Lab

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

# Environment

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

Our next step is to install the `rpy2` package.

```Python
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install rpy2
```
Now we can import the `rpy2` module into Python and set up what is called a "magic command" to run RStudio syntax from within our Python Jupyter Notebook.

```Python
# set up rpy2 magic command
%load_ext rpy2.ipython

# filter rstudio warnings
import warnings
warnings.filterwarnings('ignore')
```

Now, anytime we want to run RStudio code, we can start a code cell with the `%%R` magic command.

We may run into error messages when running the previous block of code. If needed, run the code below and replace the file path with the anaconda path to R from your local computer.

```Python
import os
os.environ['R_HOME'] = '/Users/<your user>/anaconda3/envs/<env name>/lib/R'
```

A couple more R packages to load before we start bringing in the data.
```Python
%%R
library(dplyr)
library(ggplot2)
```

1. “dplyr is a grammar of data manipulation, providing a consistent set of verbs that help you solve the most common data manipulation challenges:
  * `mutate()` adds new variables that are functions of existing variables
  * `select()` picks variables based on their names.
  * `filter()` picks cases based on their values.
  * `summarise()` reduces multiple values down to a single summary.
  * `arrange()` changes the ordering of the rows.

These all combine naturally with `group_by()` which allows you to perform any operation “by group”. You can learn more about them in [`vignette("dplyr")`](https://dplyr.tidyverse.org/articles/dplyr.html). As well as these single-table verbs, dplyr also provides a variety of two-table verbs, which you can learn about in [`vignette("two-table")`](https://dplyr.tidyverse.org/articles/two-table.html).” [Source: [dplyr.tidyverse.org](https://dplyr.tidyverse.org/)]

2. More dplyr documentation: [cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html](https://cran.r-project.org/web/packages/dplyr/vignettes/dplyr.html)

3. For more on the conceptual foundations for data transformation in `R`:
- [Chapter 5, "Data Transformation"](https://r4ds.had.co.nz/transform.html) in Hadley Wickham and Garrett Grolemund, [*R for Data Science: Visualize, Model, Transform, Tidy, and Import Data*](https://r4ds.had.co.nz/index.html) (O'Reilly, 2017).

1. “R has several systems for making graphs, but ggplot2 is one of the most elegant and most versatile. ggplot2 implements the grammar of graphics, a coherent system for describing and building graphs. With ggplot2, you can do more faster by learning one system and applying it in many places.” [[Chapter 3 “Data Visualization”](https://r4ds.had.co.nz/data-visualisation.html) in Garrett Grolemund and Hadley Wickham, *R for Data Science*]

2. “ggplot2 is a system for declaratively creating graphics, based on The Grammar of Graphics. You provide the data, tell ggplot2 how to map variables to aesthetics, what graphical primitives to use, and it takes care of the details...It’s hard to succinctly describe how ggplot2 works because it embodies a deep philosophy of visualisation. However, in most cases you start with ggplot(), supply a dataset and aesthetic mapping (with aes()). You then add on layers (like geom_point() or geom_histogram()), scales (like scale_colour_brewer()), faceting specifications (like facet_wrap()) and coordinate systems (like coord_flip()).” [[ggplot2.tidyverse.org](https://ggplot2.tidyverse.org/)]

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

# Data

We're going to work with two datasets in this lab, both developed and published by the *ProPublica* team.

`compas-scores-two-years.csv`: *ProPublica* selected fields for severity of charge, number of priors, demographics, age, sex, compas scores, and whether each person was accused of a crime within two years.

```Python
%%R
# load data from CSV file
raw_data <- read.csv("compas-scores-two-years.csv")

# load data from URL
compas_two_year_scores <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years.csv")

# show data dimensions
nrow(compas_two_year_scores)
```
We can see we have 52 columns/fields, and 7,214 rows/records in this dataset.

We can also express these steps programatically in Python:
```Python
data = pd.read_csv('compas-scores-two-years.csv', index_col=0)

data.shape

data_2 = pd.read_csv('compas-scores-two-years-violent.csv', index_col=0)

data_2.shape
```

```Python
# look at the first six rows of the dataset
pd.options.display.max_columns = None # have to do this otherwise it limits the number of cols shown

data.head() 
```
Not all of the rows are useable for the first round of analysis.

The *ProPublica* team determined a number of criteria for removing missing or unusable data. 

These criteria are listed below:
- If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
- We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
- In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
- We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

To filter the data using these criteria:
```Python
%%R
df <- dplyr::select(compas_two_year_scores, age, c_charge_degree, race, age_cat, score_text, sex, priors_count, 
                    days_b_screening_arrest, decile_score, is_recid, two_year_recid, c_jail_in, c_jail_out) %>% 
        filter(days_b_screening_arrest <= 30) %>%
        filter(days_b_screening_arrest >= -30) %>%
        filter(is_recid != -1) %>%
        filter(c_charge_degree != "O") %>%
        filter(score_text != 'N/A')
nrow(df)
```

To run that filtering operation in Python, first we'll filter out those which do not have a COMPAS-scored case, as indicated by the recidivist flag `is_recid` set at -1.

```Python
filterData = data[(data['is_recid'] != -1)]

filterData.shape
```

Within the cases with a COMPAS score, we also need to check to see if we have the right offense. 

So if the charge date of a defendant's COMPAS-scored crime was not within 30 days from when the person was arrested, it's best to assume that we do not have the right offense, and remove that row.

So we will filter out rows where **days_b_screening_arrest** is over 30 or under -30:

```Python
filterData = data[(data['days_b_screening_arrest'] <= 30) & (data['days_b_screening_arrest'] >= -30)]

filterData.shape
```

The results of both filtering processes should be the same- 52 columns or fields, and 6172 observations or rows/records.

# Exploratory Data Analysis

Higher COMPAS scores are slightly correlated with a longer length of stay. 

```Python
%%R
df$length_of_stay <- as.numeric(as.Date(df$c_jail_out) - as.Date(df$c_jail_in))
cor(df$length_of_stay, df$decile_score)
```

After filtering we have the following demographic breakdown:

```Python
# summary of age range represented in the dataset
%%R
summary(df$age_cat)
```

```Python
# Python syntax for age summary
filterData.age_cat.value_counts()
```

```Python
# summary of race information represented in the dataset
%%R
summary(df$race)
```

```Python
# Python syntax for race/ethnicity summary
filterData.race.value_counts()
```

```Python
print("Black defendants: %.2f%%" %            (3175 / 6172 * 100))
print("White defendants: %.2f%%" %            (2103 / 6172 * 100))
print("Hispanic defendants: %.2f%%" %         (509  / 6172 * 100))
print("Asian defendants: %.2f%%" %            (31   / 6172 * 100))
print("Native American defendants: %.2f%%" %  (11   / 6172 * 100))
```

```Python
# summary of risk scores
%%R
summary(df$score_text)
```

```Python
# Python syntax for risk score summary
filterData.score_text.value_counts()
```

```Python
# cross tab summary of data by race/ethnicity and gender
%%R
xtabs(~ sex + race, data=df)
```

```Python
# Python syntax for the cross tab summary, using the crosstab function
# recidivsm rates by race
pd.crosstab(filterData.sex, filterData.race)
```

```Python
# summary of data by gender
%%R
summary(df$sex)
```

```Python
# python syntax for the gender breakdown
filterData.sex.value_counts()
```

```Python
print("Men: %.2f%%" %   (4997 / 6172 * 100))
print("Women: %.2f%%" % (1175 / 6172 * 100))
```

```Python
# number of rows where two_year_recid = 1
%%R
nrow(filter(df, two_year_recid == 1))
```

```Python
# percentage of rows where two_year_recid = 2
%%R
nrow(filter(df, two_year_recid == 1)) / nrow(df) * 100
```

Judges are often presented with two sets of scores from the Compas system -- one that classifies people into High, Medium and Low risk, and a corresponding decile score. 

*ProPublica*'s analysis found a clear downward trend in the decile scores as those scores increase for white defendants.

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
# cross tab data by race and decile score
%%R
xtabs(~ decile_score + race, data=df)
```

```Python
# python syntax for the cross tab
scores_by_race = pd.crosstab(filterData.race, filterData.decile_score)

scores_by_race
```

```Python
# python syntax to generate another visualization with the decile scores disaggregated by race
labels = list(scores_by_race.columns)

aa_scores = list(scores_by_race.loc["African-American"])
c_scores = list(scores_by_race.loc["Caucasian"])

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, aa_scores, width, label='African-American')
rects2 = ax.bar(x + width/2, c_scores, width, label='Caucasian')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Scores by decile and race')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
```

# Racial Bias in Compas

These visualizations suggest that *something* is going on. 

But in order to test our intution that there is a significant difference in COMPAS scores across different racial categories, we need to run a logistic regression, comparing low scores to high scores.

After filtering out unusable rows, *ProPublica*'s next step was whether there is a significant difference in Compas scores between races. 

They explored this question by changing some variables into factors, and running a logistic regression, comparing low scores to high scores.

These factor conversions were necessary because of RStudio syntax. 

The first step would be to convert the c_charge_degree, age_cat, race, sex (which are all categorical data) into factors. 

```Python
%%R
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
summary(model)
```

But we can use the Patsy API, part of the Python `statsmodels` library, to embed these transformations within the forumla.

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

model.summary()
```
This analysis found Black defendants are 45% more likely than white defendants to receive a higher score correcting for the seriousness of their crime, previous arrests, and future criminal behavior.

```Python
%%R
control <- exp(-1.52554) / (1 + exp(-1.52554))
exp(0.47721) / (1 - control + (control * exp(0.47721)))
```

Women are 19.4% more likely than men to get a higher score.

```Python
%%R
exp(0.22127) / (1 - control + (control * exp(0.22127)))
```

Most surprisingly, people under 25 are 2.5 times as likely to get a higher score as middle aged defendants.

```Python
%%R
exp(1.30839) / (1 - control + (control * exp(1.30839)))
```

# Risk of Violent Recidivism

Compas also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score. 

The *ProPublica* team used a logistic regression to test for racial bias.

They used the second dataset `compas-scores-two-years-violent` to analyze the violent recidivism score.

```Python
%%R
# load data from CSV file
raw_data <- read.csv("compas-scores-two-years-violent.csv")

# load data from URL
compas_two_year_scores_violent <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/compas-scores-two-years-violent.csv")

# show data dimensions
nrow(compas_two_year_scores_violent)
```

Again, we can see that we have 52 columns/fields, and 7,214 rows/records in this dataset.

As before, the *ProPublica* team determined a number of criteria for removing missing or unusable data. 

These criteria are listed below:
- If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
- We coded the recidivist flag -- `is_recid` -- to be -1 if we could not find a compas case at all.
- In a similar vein, ordinary traffic offenses -- those with a `c_charge_degree` of 'O' -- will not result in Jail time are removed (only two of them).
- We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.

To filter the data using these criteria:
```Python
%%R
df1 <- dplyr::select(compas_two_year_scores_violent, age, c_charge_degree, race, age_cat, v_score_text, sex, priors_count, 
                    days_b_screening_arrest, v_decile_score, is_recid, two_year_recid) %>% 
        filter(days_b_screening_arrest <= 30) %>%
        filter(days_b_screening_arrest >= -30) %>% 
        filter(is_recid != -1) %>%
        filter(c_charge_degree != "O") %>%
        filter(v_score_text != 'N/A')
nrow(df1)
```

# More Exploratory Data Analysis

```Python
# age distribution
%%R
summary(df1$age_cat)
```

```Python
# race/ethnicity distribution
%%R
summary(df1$race)
```

```Python
# number of rows where two_year_recid = 1
%%R
nrow(filter(df1, two_year_recid == 1))
```

```Python
# percentage of rows where two_year_recid = 2
%%R
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
Agin, the *ProPublica* team explored the question of racial bias by changing some variables into factors, and running a logistic regression, comparing low scores to high scores.

```Python
%%R
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
summary(model)
```

The violent score overpredicts recidivism for black defendants by 77.3% compared to white defendants.

```Python
%%R
control <- exp(-2.24274) / (1 + exp(-2.24274))
exp(0.65893) / (1 - control + (control * exp(0.65893)))
```

Defendands under 25 are 7.4 times as likely to get a higher score as middle aged defendants.

```Python
%%R
exp(3.14591) / (1 - control + (control * exp(3.14591)))
```

# Predictive Accuracy of COMPAS

To test whether Compas scores do an accurate job of deciding whether an offender is Low, Medium or High risk,  *ProPublica* ran a Cox Proportional Hazards model. 

Northpointe, the company that created COMPAS and markets it to Law Enforcement, also ran a Cox model in their [validation study](http://cjb.sagepub.com/content/36/1/21.abstract).

*ProPublica* used the counting model and removed people when they were incarcerated. 

Due to errors in the underlying jail data, they ended up filtering out 32 rows that have an end date more than the start date. 

They determined that since there are 13,334 total rows in the data, such a small amount of errors would not affect the results.

## Proportional hazards model

What is a Cox Proportional Hazards model?

"**Survival analysis** is a branch of statistics for analyzing the expected duration of time until one or more events happen, such as death in biological organisms and failure in mechanical systems. This topic is called reliability theory or reliability analysis in engineering, duration analysis or duration modelling in economics, and event history analysis in sociology. Survival analysis attempts to answer certain questions, such as what is the proportion of a population which will survive past a certain time? Of those that survive, at what rate will they die or fail? Can multiple causes of death or failure be taken into account? How do particular circumstances or characteristics increase or decrease the probability of survival?" ([Wikipedia](https://en.wikipedia.org/wiki/Survival_analysis))

"**Proportional hazards models** are a class of survival models in statistics. Survival models relate the time that passes, before some event occurs, to one or more covariates that may be associated with that quantity of time. In a proportional hazards model, the unique effect of a unit increase in a covariate is multiplicative with respect to the hazard rate. For example, taking a drug may halve one's hazard rate for a stroke occurring, or, changing the material from which a manufactured component is constructed may double its hazard rate for failure" ([Wikipedia](https://en.wikipedia.org/wiki/Proportional_hazards_model)).

The Cox Proportional Hazards model was developed by British statistician Sir David Cox in the 1970s. 
- For more background on the model: D.R. Cox, "[Regression Models and Life Tables]( http://www.jstor.org.proxy.library.nd.edu?url=https://www.jstor.org/stable/2985181)" *Journal of the Royal Statistical Society* 34:2 (1972): 187-220.

The model is as regression model most often used to determine the association or relationship between patient survival time and predictor variables.
- [For more information on the underlying math in the Cox model](https://en.wikipedia.org/wiki/Proportional_hazards_model#The_Cox_model)
- [Examples of the Cox model used to analyze health outcomes](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Survival/BS704_Survival6.html)


## Running the model

To run this model, we need a couple of additional R packages.

The `ggfortify` package takes a curve and points object and converts it to a data frame that can be plotted using `ggplot2`.
- [For more on `ggfortify`](https://cran.r-project.org/web/packages/ggfortify/index.html)

The `survival` package contains the definition for the Cox model (as well as other statistical models).
- [For more on `survival`](https://cran.r-project.org/web/packages/survival/index.html)

```Python
%%R
library(survival)
library(ggfortify)
```

We also need to load data structured for the model.

```Python
%%R
# load data from CSV file
cox_parsed <- read.csv("cox-parsed.csv")

# load data from URL
cox_parsed <- read.csv("https://raw.githubusercontent.com/kwaldenphd/propublica-compas-lab/main/data/cox-parsed.csv")

# show data dimensions
nrow(cox_parsed)
```

We can see this data has 13,419 observations.

The next step taken by the ProPublica team was filtering the data 

```Python
%%R

cox_parsed <- filter(filter(read.csv("cox-parsed.csv"), score_text != "N/A"), end > start) %>%
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

grp <- cox_parsed[!duplicated(cox_parsed$id),]
nrow(grp)
```

The results of that filtering is 10,314 observations.

```Python
# score ranges
%%R
summary(grp$score_factor)
```

```Python
# data summary by race/ethnicity
%%R
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

People placed in the High category are 3.5 times as likely to recidivate, and the COMPAS system's concordance 63.6%. This is lower than the accuracy quoted in the Northpoint study of 68%.

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

```Python
%%R

# score factor
f2 <- Surv(start, end, event, type="counting") ~ race_factor + score_factor + race_factor * score_factor

# create model
model <- coxph(f2, data=cox_parsed)

# show model summary
print(summary(model))
```

The *ProPublica* team found that the interaction term shows a similar disparity as the logistic regression above.

They also found high risk white defendants are 3.61 more likely than low risk white defendants, while High risk black defendants are 2.99 more likely than low.

```Python
import math
print("Black High Hazard: %.2f" % (math.exp(-0.18976 + 1.28350)))
print("White High Hazard: %.2f" % (math.exp(1.28350)))
print("Black Medium Hazard: %.2f" % (math.exp(0.84286-0.17261)))
print("White Medium Hazard: %.2f" % (math.exp(0.84286)))
```

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

The *ProPublica* team found that in this case, there isn't a significant coefficient on African American's with High Scores.

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

# Directions of the Racial Bias

*ProPublica*'s analysis found that the Compas algorithm does overpredict African-American defendant's future recidivism.

The next section of the lab looks at how they explored the direction of the bias.

Their analysis found fine differences in overprediction and underprediction by comparing Compas scores across racial lines.

First, we need to load the `truth_tables.py` file with named functions created by the *ProPublica* team.

Save this file and upload to the same folder as this Jupyter Notebook.

Then, we can import functions from this file using `from truth_tables import...`.

```Python
from truth_tables import PeekyReader, Person, table, is_race, count, vtable, hightable, vhightable
from csv import DictReader

people = []
with open("cox-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                people.append(p)
    except StopIteration:
        pass

pop = list(filter(lambda i: ((i.recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.score_valid, people))))
recid = list(filter(lambda i: i.recidivist == True and i.lifetime <= 730, pop))
rset = set(recid)
surv = [i for i in pop if i not in rset]
```

```Python
print("All defendants")
table(list(recid), list(surv))
```

```Python
print("Total pop: %i" % (2681 + 1282 + 1216 + 2035))
```

```Python
import statistics
print("Average followup time %.2f (sd %.2f)" % (statistics.mean(map(lambda i: i.lifetime, pop)),
                                                statistics.stdev(map(lambda i: i.lifetime, pop))))
print("Median followup time %i" % (statistics.median(map(lambda i: i.lifetime, pop))))
```

Overall, the false positive rate is 32.35%.

```Python
print("Black defendants")
is_afam = is_race("African-American")
table(list(filter(is_afam, recid)), list(filter(is_afam, surv)))
```

That number is higher for African Americans at 44.85%.

```Python
print("White defendants")
is_white = is_race("Caucasian")
table(list(filter(is_white, recid)), list(filter(is_white, surv)))
```

And lower for whites at 23.45%.

```Python
44.85 / 23.45
```

In the *ProPublica* team's analysis, these results mean under COMPAS black defendants are 91% more likely to get a higher score and not go on to commit more crimes than white defendants after two years.

They also found that COMPAS scores misclassify white reoffenders as low risk at 70.4% more often than black reoffenders.

```Python
47.72 / 27.99
```

```Python
hightable(list(filter(is_white, recid)), list(filter(is_white, surv)))
```

```Python
hightable(list(filter(is_afam, recid)), list(filter(is_afam, surv)))
```

# Risk of Violent Recidivism

Compas also offers a score that aims to measure a persons risk of violent recidivism, which has a similar overall accuracy to the Recidivism score.

```Python
vpeople = []
with open("cox-violent-parsed.csv") as f:
    reader = PeekyReader(DictReader(f))
    try:
        while True:
            p = Person(reader)
            if p.valid:
                vpeople.append(p)
    except StopIteration:
        pass

vpop = list(filter(lambda i: ((i.violent_recidivist == True and i.lifetime <= 730) or
                              i.lifetime > 730), list(filter(lambda x: x.vscore_valid, vpeople))))
vrecid = list(filter(lambda i: i.violent_recidivist == True and i.lifetime <= 730, vpeople))
vrset = set(vrecid)
vsurv = [i for i in vpop if i not in vrset]
```

```Python
print("All defendants")
vtable(list(vrecid), list(vsurv))
```

The *ProPublica* team found these trends were further exacerbated for Black defendants.

```Python
print("Black defendants")
is_afam = is_race("African-American")
vtable(list(filter(is_afam, vrecid)), list(filter(is_afam, vsurv)))
```

```Python
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

# Gender differences in Compas scores

The *ProPublica* team used gender-specific Kaplan Meier estimates to look at differences between men and women in terms of underlying recidivism rates.

```Python
%%R

female <- filter(cox_parsed, sex == "Female")
male   <- filter(cox_parsed, sex == "Male")
male_fit <- survfit(f, data=male)
female_fit <- survfit(f, data=female)
```

```Python
%%R
summary(male_fit, times=c(730))
```

```Python
%%R
summary(female_fit, times=c(730))
```

```Python
%%R 
plotty(female_fit, "Female")
```

```Python
%%R 

plotty(male_fit, "Male")
```

From these plots, the *ProPublica* team determined the Compas score treats a High risk women the same as a Medium risk man.

# Putting It All Together

Let's take a step back here and think about what these results mean, or at least what they mean in relation to the conclusions drawn in the *ProPublica* article.

Discussion questions:
- What conclusions would you draw about the COMPAS algorithm based on this analysis?
- What other types of questions would you want to ask or what else do you want to know about the algorithm?
- How does the analysis outlined in this lab and the methodology article relate to the arguments/conclusions presented in the "Machine Bias" article?
- Limitations or shortcomings of this analysis (the analysis itself, the statistical models used, the input data, etc.)?
- Other thoughts/questions/comments/observations

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

# Lab Notebook Questions

