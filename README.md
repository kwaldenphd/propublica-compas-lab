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

MORE ON DPLYR AND GGPLOT PACKAGES

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
# summary of race information represented in the dataset
%%R
summary(df$race)
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
# cross tab summary of data by race/ethnicity and gender
%%R
xtabs(~ sex + race, data=df)
```

```Python
# summary of data by gender
%%R
summary(df$sex)
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

# Racial Bias in Compas

After filtering out unusable rows, *ProPublica*'s next step was whether there is a significant difference in Compas scores between races. 

They explored this question by changing some variables into factors, and running a logistic regression, comparing low scores to high scores.

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

```Python
%%R
library(survival)
library(ggfortify)

data <- filter(filter(read.csv("cox-parsed.csv"), score_text != "N/A"), end > start) %>%
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

grp <- data[!duplicated(data$id),]
nrow(grp)
```
# Lab Notebook Questions
