# Predicting Voluntary Employee Attrition 

### Background and Objectives

From my experience as an entrepreneur, it is critical to manage voluntary employee attrition to ensure your business remains profitable and maintains high employee morale. Specifically, I want to gain more insight as to what the key contributing factors are when an employee leaves a company voluntarily. In order to analyze these factors I am going to utilize the ‘IBM HR Analytics Employee Attrition & Performance’ dataset and build machine learning algorithms to predict which employees left voluntarily and why. The machine learning algorithms that I plan to build for this analysis are: K-Nearest Neighbors, Decision Tree, and Random Forest. I will also be plotting some ROC curves in order to evaluate which model(s) and thresholds perform the best. Then, I will make some recommendations based on my analysis as to how to better manage employee attrition. 

### Initial Questions and Assumptions

#### Business Questions
* What factors are contributing the most to employee attrition?
* Which department is suffering the most from employee attrition?
* How much does commuting time affect employee attrition?
* Approximately how much money can these models save a business?
* 
* 
* 


#### Initial Assumptions
* Working overtime will correlate strongly with employees leaving voluntarily
* Younger employees in general will be at higher risk of leaving voluntarily (<40)
* Low 'Job Satisfaction' and 'Work Life Balance' will cause employees to leave the company voluntariliy
* Employees that live far away from the office (long commute) are more susceptible to employee attrition
* 

### Raw Data Review
* total of 1470 rows (employees) and 35 columns (features)
* No null values included in this dataset
* Need to one-hot encode some of the features (BusinessTravel, Attrition (target), Gender, MaritalStatus?, OverTime, 
* What is the difference between JobInvolvement and JobLevel? 
* 

### Data Cleaning
* remove columns employee_count, employee_number, over18, standard_hours --- DONE
* remove hourly_rate, monthly_rate or daily_rate as they are directly correlated

### Exploratory Data Analysis

* In this dataset, there is a large imbalance between employees that have left and those that have stayed employed (only 237 have left the company while 1233 remained with the company ~84%) so I will use the oversampling technique SMOTE to account for this imbalance
* 
* 

### One-Hot Encoding / Elimination of Features?

* Categorical Features that require one-hot encoding: Business Travel, Department, Education Field, Job Role, Marital Status
* 

### Machine Learning Algorithms (INSERT HEADER FOR EACH ML ALGORITHM)

1) Random Forest

2) K-Nearest Neighbors / XG Boost?

3) Decision Tree? Confusion Matrix (are we looking for Precision or Recall or what and why?)

4) Regularized Linear Regression (L2) 

5) LASSO Regression (L1) 

6) Feature Importance Graph

7) ROC / AUC Curve

8) Neural Net?


### Conclusion and Recommendations

### Future Work
* Creating some sort of time series ML algorithm when the information changes over time
* Creating new features to train ML algorithms on (i.e. creating 'Young and Underpaid Feature')
