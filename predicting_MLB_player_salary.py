
# coding: utf-8

# # Predicting Major League Baseball (MLB) Player Salaries

# ## I.  Introduction

# This project builds a model using XGBoost to predict Major League Baseball (MLB) player salaries.  The model developed here is for fielding players with batting statistics.  Pitchers can also be modeled in a similar fashion, but the focus here is on predicting salaries using batting statistics. Several models are developed as we move from OLS regression to XGBoost.  As various models are developed, using better features and better prediction methods, the adjusted R-squared value increases considerably.  The baseline OLS regression model produces an R-squared value of 0.45, but with better features tops out at 0.79.  Both XGBoost models are able to push the R-squared value up to around 0.9.  One of these XGBoost models in particular represents an intuitive understanding of the factors that are driving player salaries.
# 
# Being able to predict MLB player salaries would help in determining a player's value as well as provide information about what factors drive value creation.  This would help in salary negotiations, determining team budgets, and finding out which players may be under- or overvalued.
# 
# All data for this project come from the Lahman Baseball Database.
# 
# The client for this project is the MLB teams and their organizations.  

# ## II. Data

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import explained_variance_score

import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

plt.style.use('seaborn') # pretty matplotlib plots
pd.set_option('display.width', 700)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

plt.rcParams['figure.dpi'] = 115
plt.rc('font', size=12)
plt.rc('figure', titlesize=16)
plt.rc('axes', labelsize=13)
plt.rc('axes', titlesize=16)

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# First, we import the data that we will use for this analysis.  The batting, pitching, salary, and all star data come from the Lahman Baseball Database (http://www.seanlahman.com/baseball-archive/statistics/). The consumer price index data comes from the Bureau of Labor Statistics (https://www.bls.gov/cpi/).

# In[8]:


batting = pd.read_csv('Batting.csv')
pitching = pd.read_csv('Pitching.csv')
salaries = pd.read_csv('Salaries.csv')
all_star_full = pd.read_csv('AllStarFull.csv')
cpi = pd.read_csv('CPI.csv')


# ### Batting Data

# In[9]:


batting.head()


# In[10]:


batting.describe()


# #### Batting features and their descriptions - http://m.mlb.com/glossary/
# * playerID - Player ID code
# * yearID - Year
# * stint - player’s stint: Order of appearances within a season
# * teamID - Team, a factor
# * lgID - League, a factor with levels AA AL FL NL PL UA
# * G (Games Played) - A player is credited with having played a game if he appears in it at any point -- be it as a starter or a replacement.
# * AB (At-bat) - An official at-bat comes when a batter reaches base via a fielder's choice, hit or an error (not including catcher's interference) or when a batter is put out on a non-sacrifice.
# * R (Run) - A player is awarded a run if he crosses the plate to score his team a run.
# * H (Hit) - A hit occurs when a batter strikes the baseball into fair territory and reaches base without doing so via an error or a fielder's choice.
# * 2B (Double) - A batter is credited with a double when he hits the ball into play and reaches second base without the help of an intervening error or attempt to put out another baserunner.
# * 3B (Triple) - A triple occurs when a batter hits the ball into play and reaches third base without the help of an intervening error or attempt to put out another baserunner.
# * HR (Home Run) - A home run occurs when a batter hits a fair ball and scores on the play without being put out or without the benefit of an error.
# * RBI (Runs Batted In) - A batter is credited with an RBI in most cases where the result of his plate appearance is a run being scored.
# * SB (Stolen Bases) - A stolen base occurs when a baserunner advances by taking a base to which he isn't entitled.
# * CS (Caught Stealing) - A caught stealing occurs when a runner attempts to steal but is tagged out before reaching second base, third base or home plate.
# * BB (Walk) - A walk occurs when a pitcher throws four pitches out of the strike zone, none of which are swung at by the hitter. The batter is awarded first base.
# * SO (Strikeout) - A strikeout occurs when a pitcher throws any combination of three swinging or looking strikes to a hitter.
# * IBB (Intentional Walk) - An intentional walk occurs when the defending team elects to walk a batter on purpose, putting him on first base instead of letting him try to hit.
# * HBP (Hit-by-pitch) - A hit-by-pitch occurs when a batter is struck by a pitched ball without swinging at it.
# * SH (Sacrifice Bunt) - A sacrifice bunt occurs when a player is successful in his attempt to advance a runner (or multiple runners) at least one base with a bunt.
# * SF (Sacrifice Fly) - A sacrifice fly occurs when a batter hits a fly-ball out to the outfield or foul territory that allows a runner to score.
# * GIDP (Ground Into Double Play) - A GIDP occurs when a player hits a ground ball that results in multiple outs on the bases.
# 
# 
# 

# ### Salary Data

# In[11]:


salaries.head()


# In[12]:


salaries.describe()


# ### All Star Data

# In[13]:


all_star_full.head()


# ## III. Data Wrangling

# ### Step 1:  Remove the pitchers from the batting data.
# There are pitchers in the batting data set which need to be removed.  The pitchers have very limited batting stats, so it looks like the pitchers earn a salary with out being productive at the plate.   Instead, their salary is tied to pitcher productivity and not batting productivity.

# In[14]:


pitchers = np.unique(pitching.playerID)
pitchers = pd.DataFrame(pitchers)
pitchers.columns = ['playerID']

all_df =pd.merge(batting, pitchers, how='outer', on='playerID', indicator=True)
batting_only = all_df[all_df['_merge'] == 'left_only']
batting_only.describe()


# ### Step 2: Remove all years before 1985 from the batting data.

# We need to drop the years before 1985 because we do not have salary data before then.

# In[15]:


batting_1985 = batting_only[batting_only.yearID > 1984]
batting_1985.describe()


#   

# ### Step 3: Merge the batting data with the salary data.

# Next, we merge the batting data with salary data using playerID as the common value for both data frames.

# In[16]:


df = pd.merge(batting_1985, salaries)
df.describe()


# ### Step 4: Remove data where salary is below the minimum salary in 1985.

# The minimum salary in 1985 was $60,000. We want to remove any salaries that are below this.

# In[17]:


df = df[df.salary >= 60000]
df['min_salary'] = df['salary'].groupby(df['yearID']).transform('min')

df['is_min'] = df.salary - df.min_salary

#df = df.query('is_min > 0')
df = df.query('AB > 0') # otherwise AVG cannot be computed
df.describe()


# ### Step 5: Add Experience feature and All Star feature.

# Creating a experience variable which represents years in the league.  This will also serve as a timetrend for each player as well. 

# In[18]:


df['EXP'] = df.groupby('playerID').cumcount()+1
df.sort_values(by=['playerID', 'yearID']).head(10)


# Let's also create a dummy variable which represents whether a player was an all-star or not. It will be interesting to compare the salary distributions across all-star and non-all-star players.  It will be also interesting to compare the differences of salary growth among these two groups.  Let's first inspect the all-star data.

# In[19]:


all_star_full['allStar'] = 1
all_star = all_star_full[['playerID', 'yearID', 'allStar']]
df = pd.merge(df, all_star, how='left', on=['playerID','yearID'])
df.head()


# We can see from above that there are NaNs in the allStar column.  We need to change the NaNs to zero to accuratley reflect non-all star status for a player in a given year.  The ones in this column represent that a player was an all star for a given year.

# In[20]:


df=df.fillna({'allStar':0})
df.head()


# ### Step 6: Adjust salary for inflation.

# Okay, now lets adjust salary for inflation.  For ease of interpretation, let's use 2016 dollars.  We use the consumer price index (CPI) to calculate this.

# Merge the salary data and cpi data by year.  Use the CPI value to adjust salary to 2016 dollars.

# In[21]:


salary_adj = pd.merge(df, cpi, how='left', on='yearID')
salary_adj['salary2016'] = (240/salary_adj.CPI)*salary_adj.salary
salary_adj['min_salary2016'] =(240/salary_adj.CPI)*salary_adj.min_salary
salary_adj.head()


# ## IV.  Exploratory Data Analysis

# Let's take a deeper look into our data.  
# 
# Lets look at the distributions of the target and feature variables.

# In[22]:


plt.subplot(2,2,1)
sns.kdeplot(df.G, shade=True, color="b")
plt.title("PDF of Games Played")

plt.subplot(2,2,2)
sns.kdeplot(df.AB, shade=True, color="b")
plt.title("PDF of At-Bats")

plt.subplot(2,2,3)
sns.kdeplot(df.R, shade=True, color="b")
plt.title("PDF of Runs Scored")

plt.subplot(2,2,4)
sns.kdeplot(df.H, shade=True, color="b")
plt.title("PDF of Hits")

plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
sns.kdeplot(df['2B'], shade=True, color="b")
plt.title("PDF of Doubles")

plt.subplot(2,2,2)
sns.kdeplot(df['3B'], shade=True, color="b")
plt.title("PDF of Triples")

plt.subplot(2,2,3)
sns.kdeplot(df.HR, shade=True, color="b")
plt.title("PDF of Home Runs")

plt.subplot(2,2,4)
sns.kdeplot(df.RBI, shade=True, color="b")
plt.title("PDF of Runs Batted In")

plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
sns.kdeplot(df.SB, shade=True, color="b")
plt.title("PDF of Stolen Bases")

plt.subplot(2,2,2)
sns.kdeplot(df.CS, shade=True, color="b")
plt.title("PDF of Caught Stealing")

plt.subplot(2,2,3)
sns.kdeplot(df.BB, shade=True, color="b")
plt.title("PDF of Walks")

plt.subplot(2,2,4)
sns.kdeplot(df.SO, shade=True, color="b")
plt.title("PDF of Strikeouts")

plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
sns.kdeplot(df.IBB, shade=True, color="b")
plt.title("PDF of Intentional Walks")

plt.subplot(2,2,2)
sns.kdeplot(df.HBP, shade=True, color="b")
plt.title("PDF of Hit By Pitch")

plt.subplot(2,2,3)
sns.kdeplot(df.SH, shade=True, color="b")
plt.title("PDF of Sacrifice Hits")

plt.subplot(2,2,4)
sns.kdeplot(df.SF, shade=True, color="b")
plt.title("PDF of Sacrifice Flies")

plt.tight_layout()
plt.show()

plt.subplot(1,2,1)
sns.kdeplot(df.GIDP, shade=True, color="b")
plt.title("PDF of Grounded Into Double Plays")

plt.subplot(1,2,2)
sns.kdeplot(df.EXP, shade=True, color="b")
plt.title("PDF of EXP")

plt.tight_layout()
plt.show()


# Now let's look at the distribution of the target variable. We have already adjusted salary for inflation, but let's first look at the distibution of unadjusted salary, or salary in nominal terms.  Let's also see what the growth of unadjusted salary is doing over time.

# In[23]:


sns.kdeplot(df.salary, shade=True, color="b")
plt.title("Probability Density of Unadjusted Salary")
plt.xlabel("Salary ($10 millions)")
plt.show()

sns.regplot(x='yearID',
            y='salary',
           data=df)
plt.title(' Unadjusted Salary (1985 - 2016)')
plt.xlabel('Year')
plt.ylabel('Salary ($10 millions)')
plt.show()

sns.boxplot(x="yearID", y="salary", data=df)
plt.title(' Boxplot of Unadjusted Salary (1985 - 2016)')
plt.xlabel('Year')
plt.ylabel('Salary ($10 millions)')
plt.xticks(rotation=45)
plt.show()


# Now, let's look at the distribution and scatter plot over time for salary in constant 2016 dollars.

# In[24]:


sns.kdeplot(salary_adj.salary2016, shade=True, color="b")
plt.title("Probability Density of Adjusted Salary")
plt.xlabel("")
plt.show()

sns.regplot(x='yearID',
            y='salary2016',
           data=salary_adj)
plt.title(' Adjusted Salary (1985 - 2016)')
plt.xlabel('Year')
plt.ylabel('Salary ($10 millions)')
plt.show()

sns.boxplot(x="yearID", y="salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary (1985-2016)')
plt.xlabel('Year')
plt.ylabel('Salary ($10 millions)')
plt.xticks(rotation=45)
plt.show()


# Considering the salary data is heavily skewed to the right, we will want to use the log of salary instead.  Let's look at that ditribution and scatter plot.

# In[25]:


salary_adj['log_salary2016'] = np.log(salary_adj.salary2016)


# In[26]:


sns.kdeplot(salary_adj.log_salary2016, shade=True, color="b")
plt.title("Probability Density of Adjusted Salary")
plt.xlabel("log(Salary)")
plt.show()

sns.regplot(x='yearID',
            y='log_salary2016',
           data=salary_adj)
plt.title(' Adjusted Salary (1985 - 2016)')
plt.xlabel('Year')
plt.ylabel('log(Salary)')
plt.show()

sns.boxplot(x="yearID", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary (1985-2016)')
plt.xlabel('Year')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()


# Let's look at some plots between log(Salary) and what the MLB calls "standard stats". These standard stats are made up of batting average (AVG), home runs (HR), runs batted in (RBI), runs scored (R), and stolen bases (SB).  First, let's create the batting average feature, which is simply a player's hits divided by his total at-bats for a number between zero (shown as .000) and one (shown as 1.000).
# 

# In[27]:


salary_adj['AVG'] = salary_adj.H / salary_adj.AB *1000
salary_adj.describe()


# In[28]:


sns.regplot(x="AVG", y="log_salary2016", data=salary_adj)
plt.title(' Adjusted Salary vs. Batting Average')
plt.xlabel('Batting Average')
plt.ylabel('log(Salary)')
plt.show()

sns.boxplot(x="HR", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Home Runs')
plt.xlabel('Home Runs (HR)')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.tick_params(labelsize=10)
ax = plt.axes()
plt.show()

sns.boxplot(x="RBI", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Runs Batted In')
plt.xlabel('Runs Batted In (RBI)')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x="R", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Runs Scored')
plt.xlabel('Runs Scored (R)')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x="SB", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Stolen Bases')
plt.xlabel('Stolen Bases (SB)')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x="2B", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Doubles')
plt.xlabel('Doubles (DB)')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()


# In[29]:


cols = ['log_salary2016', 'G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SF', 'SH',
               'GIDP', 'AVG']
corr = salary_adj[cols].corr()
corr.style.background_gradient().set_precision(2)


# In[30]:


sns.lmplot(x='yearID',
            y='log_salary2016',
            hue = 'allStar',
            data=salary_adj)
plt.title('All Star vs. Non-All Star')
plt.ylabel('log(Salary)')
plt.xlabel('Year')
plt.show()

sns.boxplot(x="allStar", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. All Star Status')
plt.xlabel('All Star = 1.0')
plt.ylabel('log(Salary)')
plt.xticks(rotation=45)
plt.show()


# In[31]:


sns.regplot(x='yearID',
            y='min_salary2016',
           data=salary_adj)
plt.title(' Minimum Salary vs. Year (1985 - 2016)')
plt.xlabel('Year')
plt.ylabel('Minimum Salary (2016 dollars)')
plt.show()


# In[32]:


top_50_salary = salary_adj.nlargest(170, 'salary2016')
top_50_paid = top_50_salary.playerID.unique()
top_50_paid


# In[33]:


top_50_paid_players = salary_adj[salary_adj.playerID.isin(['rodrial01', 'ramirma02', 'cabremi01', 'wellsve01', 'giambja01',
       'bondsba01', 'delgaca01', 'howarry01', 'jeterde01', 'pujolal01',
       'teixema01', 'mauerjo01', 'canoro01', 'fieldpr01', 'hamiljo03',
       'gonzaad01', 'ramirha01', 'vaughmo01', 'reyesjo01', 'uptonju01',
       'bagweje01', 'sheffga01', 'heltoto01', 'crawfca02', 'beltrca01',
       'kempma01', 'ellsbja01', 'werthja01', 'leeca01', 'ordonma01',
       'greensh01', 'soriaal01', 'sosasa01', 'piazzmi01', 'hunteto01',
       'wrighda03', 'tulowtr01', 'choosh01', 'vottojo01', 'jonesch06',
       'bayja01', 'braunry02', 'youngmi02', 'pencehu01', 'ramirar01',
       'ethiean01', 'beltrad01', 'martivi01', 'hollima01', 'sexsori01'])]



# In[34]:


top_100_salary = salary_adj.nlargest(388, 'salary2016')
top_100_paid = top_100_salary.playerID.unique()
top_100_paid


# In[35]:


top_100_paid_players = salary_adj[salary_adj.playerID.isin(['rodrial01', 'ramirma02', 'cabremi01', 'wellsve01', 'giambja01',
       'bondsba01', 'delgaca01', 'howarry01', 'jeterde01', 'pujolal01',
       'teixema01', 'mauerjo01', 'canoro01', 'fieldpr01', 'hamiljo03',
       'gonzaad01', 'ramirha01', 'vaughmo01', 'reyesjo01', 'uptonju01',
       'bagweje01', 'sheffga01', 'heltoto01', 'crawfca02', 'beltrca01',
       'kempma01', 'ellsbja01', 'werthja01', 'leeca01', 'ordonma01',
       'greensh01', 'soriaal01', 'sosasa01', 'piazzmi01', 'hunteto01',
       'wrighda03', 'tulowtr01', 'choosh01', 'vottojo01', 'jonesch06',
       'bayja01', 'braunry02', 'youngmi02', 'pencehu01', 'ramirar01',
       'ethiean01', 'beltrad01', 'martivi01', 'hollima01', 'sexsori01',
       'belleal01', 'abreubo01', 'sandopa01', 'furcara01', 'thomeji01',
       'gonzaca01', 'guerrvl01', 'berkmla01', 'mccanbr01', 'willibe02',
       'gonzaju03', 'mondera01', 'griffke02', 'walkela01', 'utleych01',
       'poseybu01', 'tejadmi01', 'morneju01', 'jonesan01', 'jonesad01',
       'kinslia01', 'napolmi01', 'drewjd01', 'ortizda01', 'grandcu01',
       'troutmi01', 'burrepa01', 'hidalri01', 'burnije01', 'markani01',
       'rasmuco01', 'wietema01', 'peraljh01', 'damonjo01', 'matsuhi01',
       'fukudko01', 'higgibo02', 'wilsopr01', 'leede02', 'andruel01',
       'dyeje01', 'molinya01', 'martiru01', 'rowanaa01', 'mcgwima01',
       'kendaja01', 'rolensc01', 'posadjo01', 'bautijo02', 'uptonbj01'
                                                           ])]


# In[36]:


top_50_paid_players.set_index('EXP', inplace=True)
top_50_paid_players.groupby('playerID')['log_salary2016'].plot(legend=False)
plt.title("Salary Profile of Top 50 Highest Paid MLB Players")
plt.ylabel("log(Salary)")
plt.show()

top_100_paid_players.set_index('EXP', inplace=True)
top_100_paid_players.groupby('playerID')['log_salary2016'].plot(legend=False)
plt.title("Salary Profile of Top 100 Highest Paid MLB Players")
plt.ylabel("log(Salary)")
plt.show()


# In[37]:


sns.boxplot(x="EXP", y="log_salary2016", data=salary_adj)
plt.title(' Boxplot of Adjusted Salary vs. Experience')
plt.xlabel('Experience (years)')
plt.ylabel('log(Salary)')

#plt.xticks(rotation=90)
plt.show()


# ## V. Feature Engineering

# ### 1.  Create a quadratic term for experience (EXP-squared)

# Seeing that salary seems to have a non-linear relationship with salary, let's add a quadratic term for experience to the feature set.

# In[38]:


salary_adj['EXP_SQ']=np.square(salary_adj['EXP'])
#salary_adj.sort_values(by=['playerID', 'yearID'])
salary_adj.describe()


# ### 2.  Create lags of the target variable and feature set.

# Let's create lagged values of the target variable and lagged values of the features.  This is based on the idea that salary is based off past player performance, not current performance, since the salaries are set before a given season.  Let's also use past salary as a feature as well as this is the best predictor of current salary we have.

# In[39]:


# lagged values of salary
salary_adj['sal_t_1'] = salary_adj.groupby(['playerID'])['salary2016'].shift(1)
salary_adj['sal_t_2'] = salary_adj.groupby(['playerID'])['salary2016'].shift(2)
salary_adj['sal_t_3'] = salary_adj.groupby(['playerID'])['salary2016'].shift(3)

# first difference of salary lagged one period
salary_adj['sal_diff'] = salary_adj.salary2016 - salary_adj.sal_t_1
salary_adj['sal_diff_t_1'] = salary_adj.groupby(['playerID'])['sal_diff'].shift(1)

# lagged values of the features
salary_adj['G_t_1'] = salary_adj.groupby(['playerID'])['G'].shift(1)
salary_adj['G_t_2'] = salary_adj.groupby(['playerID'])['G'].shift(2)

salary_adj['AB_t_1'] = salary_adj.groupby(['playerID'])['AB'].shift(1)
salary_adj['AB_t_2'] = salary_adj.groupby(['playerID'])['AB'].shift(2)

salary_adj['R_t_1'] = salary_adj.groupby(['playerID'])['R'].shift(1)
salary_adj['R_t_2'] = salary_adj.groupby(['playerID'])['R'].shift(2)

salary_adj['H_t_1'] = salary_adj.groupby(['playerID'])['H'].shift(1)
salary_adj['H_t_2'] = salary_adj.groupby(['playerID'])['H'].shift(2)

salary_adj['2B_t_1'] = salary_adj.groupby(['playerID'])['2B'].shift(1)
salary_adj['2B_t_2'] = salary_adj.groupby(['playerID'])['2B'].shift(2)

salary_adj['3B_t_1'] = salary_adj.groupby(['playerID'])['3B'].shift(1)
salary_adj['3B_t_2'] = salary_adj.groupby(['playerID'])['3B'].shift(2)

salary_adj['HR_t_1'] = salary_adj.groupby(['playerID'])['HR'].shift(1)
salary_adj['HR_t_2'] = salary_adj.groupby(['playerID'])['HR'].shift(2)

salary_adj['RBI_t_1'] = salary_adj.groupby(['playerID'])['RBI'].shift(1)
salary_adj['RBI_t_2'] = salary_adj.groupby(['playerID'])['RBI'].shift(2)

salary_adj['AVG_t_1'] = salary_adj.groupby(['playerID'])['AVG'].shift(1)
salary_adj['AVG_t_2'] = salary_adj.groupby(['playerID'])['AVG'].shift(2)

salary_adj['SB_t_1'] = salary_adj.groupby(['playerID'])['SB'].shift(1)
salary_adj['SB_t_2'] = salary_adj.groupby(['playerID'])['SB'].shift(2)

salary_adj['CS_t_1'] = salary_adj.groupby(['playerID'])['CS'].shift(1)
salary_adj['CS_t_2'] = salary_adj.groupby(['playerID'])['CS'].shift(2)

salary_adj['BB_t_1'] = salary_adj.groupby(['playerID'])['BB'].shift(1)
salary_adj['BB_t_2'] = salary_adj.groupby(['playerID'])['BB'].shift(2)

salary_adj['SO_t_1'] = salary_adj.groupby(['playerID'])['SO'].shift(1)
salary_adj['SO_t_2'] = salary_adj.groupby(['playerID'])['SO'].shift(2)

salary_adj['IBB_t_1'] = salary_adj.groupby(['playerID'])['IBB'].shift(1)
salary_adj['IBB_t_2'] = salary_adj.groupby(['playerID'])['IBB'].shift(2)

salary_adj['HBP_t_1'] = salary_adj.groupby(['playerID'])['HBP'].shift(1)
salary_adj['HBP_t_2'] = salary_adj.groupby(['playerID'])['HBP'].shift(2)

salary_adj['SH_t_1'] = salary_adj.groupby(['playerID'])['SH'].shift(1)
salary_adj['SH_t_2'] = salary_adj.groupby(['playerID'])['SH'].shift(2)

salary_adj['SF_t_1'] = salary_adj.groupby(['playerID'])['SF'].shift(1)
salary_adj['SF_t_2'] = salary_adj.groupby(['playerID'])['SF'].shift(2)

salary_adj['GIDP_t_1'] = salary_adj.groupby(['playerID'])['GIDP'].shift(1)
salary_adj['GIDP_t_2'] = salary_adj.groupby(['playerID'])['GIDP'].shift(2)

salary_adj['allStar_t_1'] = salary_adj.groupby(['playerID'])['allStar'].shift(1)
salary_adj['allStar_t_2'] = salary_adj.groupby(['playerID'])['allStar'].shift(2)
#salary_adj.sort_values(by=['playerID', 'yearID'])


# ### 3.  Calculate on base percentage (OBP).

# On Base Percentage (aka OBP, On Base Average, OBA) is a measure of how often a batter reaches base. It is approximately equal to Times on Base/Plate appearances.
# 
# The full formula is OBP = (Hits + Walks + Hit by Pitch) / (At Bats + Walks + Hit by Pitch + Sacrifice Flies). Batters are not credited with reaching base on an error or fielder's choice, and they are not charged with an opportunity if they make a sacrifice bunt.

# In[40]:


salary_adj['OBP'] = 1000*(salary_adj.H + salary_adj.BB + salary_adj.HBP)/(salary_adj.AB + salary_adj.BB + salary_adj.HBP
                                                                    + salary_adj.SF)

# Create lagged value of OBP
salary_adj['OBP_t_1'] = salary_adj.groupby(['playerID'])['OBP'].shift(1)
salary_adj['OBP_t_2'] = salary_adj.groupby(['playerID'])['OBP'].shift(2)


# ### 4. Create interactions between certain features.

# Let's experiment by interacting some of the features.  For example to pick up the effect of a player getting better over time, or at least staying consistent at a high level, we could interact experience (EXP) with on base percentage (OBP).  Another example would be to pick up the effect of a player that both hits a lot of home runs (HR) and gets on base a lot (OBP).  We could try other interactions, but let's just stick to these two for now.

# In[41]:


salary_adj['EXP_OBP'] = salary_adj.EXP*salary_adj.OBP
salary_adj['OBP_HR'] = salary_adj.OBP*salary_adj.HR

# Create lag value of interactions above
salary_adj['EXP_OBP_t_1'] = salary_adj.groupby(['playerID'])['EXP_OBP'].shift(1)
salary_adj['EXP_OBP_t_2'] = salary_adj.groupby(['playerID'])['EXP_OBP'].shift(2)
salary_adj['OBP_HR_t_1'] = salary_adj.groupby(['playerID'])['OBP_HR'].shift(1)
salary_adj['OBP_HR_t_2'] = salary_adj.groupby(['playerID'])['OBP_HR'].shift(2)

salary_adj['constant'] = 1


# In[42]:


salary_adj.describe()


# ## VI. Modeling and Results

# In[43]:


y = salary_adj.log_salary2016
x_baseline = salary_adj[['G_t_1', 'AB_t_1', 'R_t_1', 
                'H_t_1', '2B_t_1', '3B_t_1', 'HR_t_1',
                'RBI_t_1', 'SB_t_1', 'CS_t_1', 'BB_t_1', 'SO_t_1', 
                'IBB_t_1', 'HBP_t_1', 'SH_t_1', 'SF_t_1',
                'GIDP_t_1', 'constant']]

x_lag1 = salary_adj[['sal_t_1', 'G_t_1', 'AB_t_1', 'R_t_1', 
                'H_t_1', '2B_t_1', '3B_t_1', 'HR_t_1',
                'RBI_t_1', 'SB_t_1', 'CS_t_1', 'BB_t_1', 'SO_t_1', 
                'IBB_t_1', 'HBP_t_1', 'SH_t_1', 'SF_t_1',
                'GIDP_t_1', 'AVG_t_1',
                'OBP_t_1', 'EXP', 'EXP_SQ', 'allStar_t_1', 'EXP_OBP_t_1', 'OBP_HR_t_1', 'min_salary2016', 'constant']]

x = salary_adj[['sal_t_1', 'sal_t_2', 'AB_t_1', 'AB_t_2', 'R_t_1', 'R_t_2',
                'H_t_1', 'H_t_2', '2B_t_1','2B_t_2','SO_t_1', 'SO_t_2',
                'AVG_t_1', 'AVG_t_2',
                'OBP_t_1', 'OBP_t_2', 'EXP', 'EXP_OBP_t_1', 'EXP_OBP_t_2', 'OBP_HR_t_1', 'OBP_HR_t_2', 'min_salary2016', 'constant']]


# Create the training and test splits.

# In[44]:


X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(x_baseline, y, test_size=.25, random_state=314)
X_train_lag1, X_test_lag1, y_train_lag1, y_test_lag1 = train_test_split(x_lag1, y, test_size=.25, random_state=314)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=.25, random_state=314)


# ### 1.  Linear Regression Models

# In[45]:


ols_base = sm.OLS(y_base_train, X_base_train, missing='drop')
results_ols_base = ols_base.fit()
print(results_ols_base.summary())


# In[46]:


residuals_ols_base = results_ols_base.resid
plt.scatter(results_ols_base.fittedvalues, results_ols_base.resid)
plt.title("Baseline OLS Regression: Residual Plot")
plt.xlabel("Fitted Values (log(Salary))")
plt.ylabel("Residuals")
plt.show()

sns.kdeplot(results_ols_base.resid, shade=True, color="b")
plt.title("Baseline OLS Regression: Probabilty Density of Residuals")
plt.xlabel("Residuals")
plt.show()

import pylab
import scipy.stats as scipystats
scipystats.probplot(results_ols_base.resid, dist="norm", plot=pylab)
pylab.show()


# In[47]:


ols_lag1 = sm.OLS(y_train_lag1, X_train_lag1, missing='drop')
results_ols_lag1 = ols_lag1.fit()
print(results_ols_lag1.summary())


# In[48]:


residuals_ols_lag1 = results_ols_lag1.resid
plt.scatter(results_ols_lag1.fittedvalues, results_ols_lag1.resid)
plt.title("Lagged Features OLS Regression: Residual Plot")
plt.xlabel("Fitted Values (log(Salary))")
plt.ylabel("Residuals")
plt.show()

sns.kdeplot(results_ols_lag1.resid, shade=True, color="b")
plt.title("Lagged Features OLS Regression: Probabilty Density of Residuals")
plt.xlabel("Residuals")
plt.show()

scipystats.probplot(results_ols_lag1.resid, dist="norm", plot=pylab)
pylab.show()


# ### 2.  XGBoost Model

# In[49]:



import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=5, random_state=314)

model_xgb_lag1 = XGBRegressor(objective='reg:linear',
                         n_estimators=400,
                         max_depth=6,
                         learning_rate = 0.08,
                         colsample_bytree=1,
                         subsample = .8,
                         gamma = 1,
                         min_child_weight=5,
                         nthreads=4,
                         seed=314,
                         eval_metric="rmse")



results_lag1 = cross_val_score(model_xgb_lag1, X_train_lag1, y_train_lag1, cv=kfold)

print(results_lag1)

model_xgb_lag1.fit(X_train_lag1, y_train_lag1)


# make predictions for test data
y_pred_lag1 = model_xgb_lag1.predict(X_test_lag1)

print("Score_XGB:", model_xgb_lag1.score(X_test_lag1, y_test_lag1))


# In[50]:


import graphviz
xgb.plot_tree(model_xgb_lag1, num_trees=5, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(100, 100)
fig.savefig('tree.png')


# In[51]:


residual = y_test_lag1 - y_pred_lag1
plt.scatter(y_pred_lag1, residual)
plt.xlabel('Y-hat')
plt.ylabel('Residuals')
plt.title('Lagged Features XGBoost: Residual Plot')
plt.show()

sns.kdeplot(residual, shade=True, color="b")
plt.title("Lagged Features XGBoost: Probabilty Density of Residuals")
plt.xlabel("Residuals")
plt.show()

import pylab
import scipy.stats as scipystats
scipystats.probplot(residual, dist="norm", plot=pylab)
pylab.show()


# In[52]:


xgb.plot_importance(model_xgb_lag1)


# Now let's use the feature importance graph from above to try and increase our model performance.  Let's try removing all the features that have a F score of less than 100.  Let's also add in two-year lags for the features that have an F score of above 100.  Let's see if this makes any difference in our model performance.

# In[53]:


model_xgb = XGBRegressor(objective='reg:linear',
                         n_estimators=400,
                         max_depth=6,
                         learning_rate = 0.08,
                         colsample_bytree=1,
                         subsample = .9,
                         gamma = 1,
                         min_child_weight=5,
                         nthreads=4,
                         seed=314,
                         eval_metric="rmse")

results = cross_val_score(model_xgb, X_train, y_train, cv=kfold)
print(results)
model_xgb.fit(X_train, y_train)

# make predictions for test data
y_pred = model_xgb.predict(X_test)
print("R-squared for XGBoost:", model_xgb.score(X_test, y_test))


# In[54]:


residual_2 = y_test - y_pred
plt.scatter(y_pred, residual)
plt.xlabel('Y-hat')
plt.ylabel('Residuals')
plt.title("Adjusted XGBoost: Residual Plot")
plt.show()

sns.kdeplot(residual_2, shade=True, color ="b")
plt.title("Adjusted XGBoost: Probabilty Density of Residuals")
plt.xlabel("Residuals")
plt.show()

import pylab
import scipy.stats as scipystats
scipystats.probplot(residual_2, dist="norm", plot=pylab)
pylab.show()


# In[55]:


xgb.plot_importance(model_xgb)

