# Predicting Voluntary Employee Attrition 

![Alt](./images/I-Quit.jpg)

### Background and Objectives

From my experience as an entrepreneur, it is critical to manage voluntary employee attrition ensuring your business remains profitable and maintains a high level of efficiency. According to Work Institute's 2017 Retention Report, it is estimated to cost as much as 33% of a worker's annual salary to replace an employee. Additionally, the report found that about 75% of the reasons employees leave could have been prevented. So, to help businesses save money and time trying to find replacements, I want to gain more insight as to what the key contributing factors are when an employee leaves a company voluntarily.

In order to analyze these factors I am going to analyze the ‘IBM HR Analytics Employee Attrition Dataset’ (ADD KAGGLE LINK) dataset and build various machine learning algorithms to predict which employees left voluntarily and why. Also, I want determine which factors have the most correlation to an employee leaving voluntairy so HR departments can better understand which employees may be at more risk of leaving. After performing this analysis, I will make some recommendations to HR Departments on actions they can take to help minimize their employee attrition rate and retain talent. 

USE PHRASE "RETAIN TALENT"

Source: https://www.hrdive.com/news/study-turnover-costs-employers-15000-per-worker/449142/
Source: https://www.benefitnews.com/news/avoidable-turnover-costing-employers-big?brief=00000152-14a7-d1cc-a5fa-7cffccf00000&utm_content=socialflow&utm_campaign=ebnmagazine&utm_source=twitter&utm_medium=social

### Initial Questions and Assumptions

#### Business Questions to Answer
* What factors are contributing the most to employee attrition?
* Which department is suffering the most from employee attrition?
* How much does commuting time affect employee attrition?
* Approximately how much money can these models save a business?

#### Initial Employee Attrition Assumptions
* Working overtime will correlate strongly with employees leaving voluntarily
* Younger employees in general will be at higher risk of leaving voluntarily (<40)
* Low 'Job Satisfaction' and 'Work Life Balance' will cause employees to leave the company voluntariliy
* Employees that live far away from the office (long commute) are more susceptible to employee attrition

### Raw Data Review

After loading the dataset in to a Pandas DataFrame in Python, here is how the data looks and my main takeaways:

|    |   Age | Attrition   | BusinessTravel    |   DailyRate | Department             |   DistanceFromHome |   Education | EducationField   |   EmployeeCount |   EmployeeNumber |   EnvironmentSatisfaction | Gender   |   HourlyRate |   JobInvolvement |   JobLevel | JobRole               |   JobSatisfaction | MaritalStatus   |   MonthlyIncome |   MonthlyRate |   NumCompaniesWorked | Over18   | OverTime   |   PercentSalaryHike |   PerformanceRating |   RelationshipSatisfaction |   StandardHours |   StockOptionLevel |   TotalWorkingYears |   TrainingTimesLastYear |   WorkLifeBalance |   YearsAtCompany |   YearsInCurrentRole |   YearsSinceLastPromotion |   YearsWithCurrManager |
|---:|------:|:------------|:------------------|------------:|:-----------------------|-------------------:|------------:|:-----------------|----------------:|-----------------:|--------------------------:|:---------|-------------:|-----------------:|-----------:|:----------------------|------------------:|:----------------|----------------:|--------------:|---------------------:|:---------|:-----------|--------------------:|--------------------:|---------------------------:|----------------:|-------------------:|--------------------:|------------------------:|------------------:|-----------------:|---------------------:|--------------------------:|-----------------------:|
|  0 |    41 | Yes         | Travel_Rarely     |        1102 | Sales                  |                  1 |           2 | Life Sciences    |               1 |                1 |                         2 | Female   |           94 |                3 |          2 | Sales Executive       |                 4 | Single          |            5993 |         19479 |                    8 | Y        | Yes        |                  11 |                   3 |                          1 |              80 |                  0 |                   8 |                       0 |                 1 |                6 |                    4 |                         0 |                      5 |
|  1 |    49 | No          | Travel_Frequently |         279 | Research & Development |                  8 |           1 | Life Sciences    |               1 |                2 |                         3 | Male     |           61 |                2 |          2 | Research Scientist    |                 2 | Married         |            5130 |         24907 |                    1 | Y        | No         |                  23 |                   4 |                          4 |              80 |                  1 |                  10 |                       3 |                 3 |               10 |                    7 |                         1 |                      7 |
|  2 |    37 | Yes         | Travel_Rarely     |        1373 | Research & Development |                  2 |           2 | Other            |               1 |                4 |                         4 | Male     |           92 |                2 |          1 | Laboratory Technician |                 3 | Single          |            2090 |          2396 |                    6 | Y        | Yes        |                  15 |                   3 |                          2 |              80 |                  0 |                   7 |                       3 |                 3 |                0 |                    0 |                         0 |                      0 |
|  3 |    33 | No          | Travel_Frequently |        1392 | Research & Development |                  3 |           4 | Life Sciences    |               1 |                5 |                         4 | Female   |           56 |                3 |          1 | Research Scientist    |                 3 | Married         |            2909 |         23159 |                    1 | Y        | Yes        |                  11 |                   3 |                          3 |              80 |                  0 |                   8 |                       3 |                 3 |                8 |                    7 |                         3 |                      0 |
|  4 |    27 | No          | Travel_Rarely     |         591 | Research & Development |                  2 |           1 | Medical          |               1 |                7 |                         1 | Male     |           40 |                3 |          1 | Laboratory Technician |                 2 | Married         |            3468 |         16632 |                    9 | Y        | No         |                  12 |                   3 |                          4 |              80 |                  1 |                   6 |                       3 |                 3 |                2 |                    2 |                         2 |                      2 |

* Dataset includes a total of 1470 rows (i.e. Employees) and 35 columns (i.e. Features)
* There are several columns that include categorical values: BusinessTravel, Department, EducationField, JobRole, and Marital Status
* The Dataset does not include any 'Null' or 'NaN' values
* There are some columns that can be dropped from the dataset as they do not provide any value: EmployeeCount, EmployeeNumber, StandardHours and Over18
* Two columns will need to be converted from "Yes" or "No" values to binary values: Attrition (our 'target') and OverTime

### Data Cleaning

As stated above, there were no 'Null' or 'NaN' values in the dataset, so I did not need to drop any rows which is beneficial since there are only 1470 total entries. Next, I wanted to reformat the column names to make them more "Pythonic" so I ran a script to make the features all lowercase and add an 'underscore' separating the words (i.e. 'BusinessTravel' was converted to 'business_travel'). Also after further review, it's apparent that there are some features that can be completely dropped because they do not provide any value. These columns included: "employee_count", "employee_number", "over18, "standard_hours". 

### Exploratory Data Analysis

After the dataset was cleaned, I wanted to perform some Exploratory Data Analysis (EDA) on the features to confirm my initial assumptions and see if any other features had strong correlations to employee attrition. So, I plotted a histogram and a 'Heat Map'of all the features to better understand the distributions and correlations of each column in the dataset. There were a few things that stuck out to me after generating the plot (INSERT PLOTS HERE AND HERE if you want to take a closer look):

**Feature Distribution Takeaways:**
* The vast majority of the company lives within 10 miles of the office.
* Job Satisfaction among the employees is high with most scores at a 3 or a 4 with a 4 being highest score.
* Majority of employees have worked at only 1 previous company before current position.
* **IMPORTANT**: The 'Attrition' target variable is highly imbalanced as only ~16% of employees left the company (~84% stayed with company). So, I will need to be sure to account for this imbalance while I am building my machine learning algorithms.

**Heat Map Takeaways:**
According to the 'Heat Map', the features that had the highest correlation with attrition were:
* 'age', 'job_level', 'monthly_income', 'total_working_years', 'stock_option_level', 'years_at_company', and 'years_with_curr_manager'

*Note: The Heat Map did not include any categorical features, so it only provides insight to the numeric features. I will address the categorical features below* 

Lastly, after plotting each feature with respect to the 'Attrition' target variable, I found these 8 features had valuable insight on if an employee voluntarily left the company.

![Alt](./images/eda_plot_8.png)

### Categorical Features and One-Hot Encoding

As noted above, there were several columns that need to be one-hot encoded or changed to a binary value so I can utilize the features in my machine learning algorithms. These features included: "business_travel", "attrition", "department", "education_field", "job_role", and "marital_status"
* Categorical Features that require one-hot encoding: Business Travel, Department, Education Field, Job Role, Marital Status
* 

### Machine Learning Algorithms (INSERT HEADER FOR EACH ML ALGORITHM)

1) Logistic Regression: Want to minimize the Recall (TP/TP+FN) and so I used my findings from EDA to chose the Top 10 Features that I believe would best correlate to an employee leaving voluntarily (see list below):

assumed_top_10 = ['age', 'job_satisfaction', 'monthly_income', 'over_time', 'work_life_balance', 'years_since_last_promotion', 'marital_status_single', 'department_sales', 'num_companies_worked', 'job_involvement']


1) Random Forest

3) Decision Tree? Confusion Matrix (are we looking for Precision or Recall or what and why?)

5) LASSO Regression (L1) 

6) Feature Importance Graph

7) ROC / AUC Curve


### Conclusion and Recommendations

### Future Work 
[] Creating some sort of time series ML algorithm when the information changes over time
[] Perform some feature engineering to create new features for training the machine learning algorithms (i.e. creating 'Young and Underpaid' feature)
[] Perform XGBoost ML model
[] Create a Flask application for HR departments to access and input in respective employee features to compute a probability of them voluntarily leaving 
