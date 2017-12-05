import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read in the batting and salary data
batting = pd.read_csv('Batting.csv')
salaries = pd.read_csv('Salaries.csv')

print(batting.head())
print(salaries.head())

# Drop years before 1985. Salary data not available.
batting_1985 = batting[batting['yearID'] > 1984]
print(batting_1985.head())

# Drop observations for number of games played less than 100
batting_1985_100g = batting_1985[batting_1985['G'] > 99]
print(batting_1985_100g.head())

# plot salary over time for the years 1985 - 2016 (unadjusted salary)
sns.regplot(x='yearID',
            y='salary',
            data=salaries)
plt.title("Salary vs. Time (1985-2016)")
plt.ylabel("Salary ($)")
plt.xlabel("Year")
plt.show()

# plot salary distribution
sns.distplot(salaries.salary)
plt.show()

# There are rows that have zeros for the salary. Remove these observations.
# The lowest minimum salary in current year dollars was in 1985.  It was
# $60,000.  Remove the observations where salary is less $60,000.
salaries_clean = salaries[salaries['salary'] != 0]
salaries_clean = salaries_clean[salaries_clean['salary'] >= 60000]
print(salaries_clean.head())

# plot salary distribution after zero salary was removed
sns.distplot(salaries_clean.salary)
plt.show()

# Need to adjust salary over time so it is in constant 2016 dollars.
# Use the CPI to calculate this.

# Read in the CPI data
cpi = pd.read_csv('CPI.csv')
print(cpi.head())

# Merge the salary data and cpi data by year
salary_adj = pd.merge(salaries_clean, cpi, how='left', on='yearID')
print(salary_adj.head())

# Use the CPI value to adjust salary to 2016 dollars.
# Change salary to 1000s of dollars.
salary_adj['salary2016'] = (240/salary_adj.CPI)*salary_adj.salary/1000
print(salary_adj.head())

# Plot salary over time in 2016 dollars.
sns.regplot(x='yearID',
            y='salary2016',
            data=salary_adj)
plt.title("Salary vs. Time (1985-2016): 2016 Dollars")
plt.ylabel("Salary ($1000s)")
plt.xlabel("Year")
plt.show()

# Detecting outliers in salary2016 values
# Group_by yearID, then find corresponding z-score for each salary

salary_adj['mean_salary'] = salary_adj['salary2016'].groupby(salary_adj['yearID']).transform('mean')
salary_adj['std_salary'] = salary_adj['salary2016'].groupby(salary_adj['yearID']).transform('std')
print(salary_adj.head())

salary_adj['z_salary'] = (salary_adj.salary2016 - salary_adj.mean_salary)/salary_adj.std_salary
print(salary_adj.head())

# plot z-scores for salary distribution
sns.distplot(salary_adj.z_salary)
plt.show()

