# Telco Churn Classification Project
 
# Project Description
 
In the competitive landscape of telecommunications, understanding the factors that drive customer churn is essential for maintaining business sustainability and growth. The "Uncovering Customer Churn Drivers at Telco" project aims to delve deep into Telco's customer data to identify the underlying causes behind customer churn. By analyzing the patterns and trends within the data, this project seeks to provide actionable insights that will enable Telco to proactively address churn and retain valuable customers.
 
# Project Goal
 
* Identify Key Churn Drivers: Determine the primary factors influencing customer churn at Telco by conducting a thorough analysis of customer data.
* Develop Predictive Models: Build accurate machine learning models that predict customer churn based on historical data.
* Provide Actionable Insights: Translate complex data analysis into actionable insights and recommendations for Telco.
* Long-Term Churn Prevention: Extend the project's impact beyond immediate insights by creating a roadmap for ongoing churn prevention.
 
# Initial Thoughts
 
* My initial hypothesis is that tenure has the strongest relationship to churn so the longer the customer stays at the company the lesslikely they are to churn.

# Initial Questions

* Does having tech support affects churn?  
* Does tenure have a direct relationship with churn?  
* Does the contract type affect churn?   
* Does internet service type have a strong relationship with churn? 

 
# The Plan
 
* - Acquire
    - Acquiring specific columns that i need for the project from the telco_churn dataset using sql.
    - Read the sql query into a dataframe.
 
* Prepare data
   * Create new columns by transforming and utilizing existing data features.
       * Drop the customer_id column
       * Check all rows in the df with no/null values and replacing them with an appropriate value.
       * Group different values in the columns and make them categorical if needed.
       * One-hot encode categorical culomns with get dummies.
       * Drop all the encoded columns that are not useful.
      
 
* Explore data in search of drivers of churn
   * Answer the following initial questions
       * Does having tech support affects churn?
       * Does tenure have a direct relationship with churn?
       * Does the contract type affect churn? 
       * Does internet service type have a strong relationship with churn?
       * Does being a senior citizen affect churn
      
* Create a predictive model to determine customer churn likelihood.
   * Utilize insights from exploratory analysis to construct predictive models of various types.
   * Assess model performance using both training and validation datasets.
   * Choose the optimal model by considering the highest accuracy.
   * Assess the chosen top-performing model using the test dataset.
 
* Draw conclusions
 
# Data Dictionary

  * This table provides a clear definition of each feature present in the dataset along with their respective descriptions.

| Feature                | Definition | Data Type |
|:-----------------------|:-----------|:----------|
| gender                 | Gender of the customer | Object |
| senior_citizen         | 0 if not a senior citizen, 1 if a senior citizen | Integer |
| partner                | Whether the customer has a partner (Yes/No) | Object |
| dependents             | Whether the customer has dependents (Yes/No) | Object |
| tenure                 | Number of months the customer has been with the company | Integer |
| phone_service          | Whether the customer has phone service (Yes/No) | Object |
| multiple_lines         | Whether the customer has multiple lines (Yes/No) | Object |
| online_security        | Whether the customer has online security (Yes/No) | Object |
| online_backup          | Whether the customer has online backup (Yes/No) | Object |
| device_protection      | Whether the customer has device protection (Yes/No) | Object |
| tech_support           | Whether the customer has tech support (Yes/No) | Object |
| streaming_tv           | Whether the customer has streaming TV (Yes/No) | Object |
| streaming_movies       | Whether the customer has streaming movies (Yes/No) | Object |
| paperless_billing      | Whether the customer has paperless billing (Yes/No) | Object |
| monthly_charges        | Monthly charges for the customer | Float |
| total_charges          | Total charges incurred by the customer | Object |
| churn                  | Whether the customer has churned (Yes/No) | Object |
| payment_type           | Method of payment for the service | Object |
| contract_type          | Type of contract (e.g., Month-to-Month, One/Two Years) | Object |
| internet_service_type  | Type of internet service (e.g., DSL, Fiber optic) | Object |


 
# Steps to Reproduce
1) Clone this repository by clicking on the SSH link.
2) If you have been granted access to the Codeup MySQL database(i.e. ask for permission from staff if you do not have access):
   i) Save the env.py file in the repository. Make sure to include your user, password, and host variables.
   ii) Add the env.py file to the .gitignore file to keep your sensitive information secure.
4) Run the Jupyter Notebook file in your local environment.
 
# Conclusions
- Having tech support greatly reducing the customers probability of churning.
- The longer the customer has been with the company the less likely they are to churn.
- New customers tend to have a high churn rate.
- Customers with a monthly plan have a higher churn rate than customers with a yearly plan.
- Customers with DSL tend to have a significantly lower churn rate compared to customers with Fiber optics.
- It is apparent that senior citizens, although a minority of customer churn, exhibit a considerably higher churn rate in comparison to non-senior citizen customers.

# Takeaways
i) Contract Type: In the decision tree model, contract type stands out as the most crucial feature with an importance score of around 0.619. It also holds significance in the random forest model, ranking second with an importance score of approximately 0.1987.

ii) Tenure: Tenure is the second most important feature in the decision tree model, with an importance score of roughly 0.194. Interestingly, it takes the lead in importance in the random forest model, where its score is approximately 0.209.

iii) Monthly Charges: Monthly charges emerge as the third most important feature in the random forest model, with an importance score of about 0.142.

iv) Internet Service Type: In the decision tree model, internet service type holds the third position in importance, with a score of roughly 0.167. It also maintains significance in the random forest model, ranking fifth with an importance score of approximately 0.113.

v) Total Charges: Total charges play a role in churn prediction and are considered the fourth most important feature in the random forest model, with an importance score of approximately 0.132.

 
# Recommendations
- Encourage customers to choose longer-term contracts, such as one or two-year contracts, as these seem to have a positive impact on customer retention.
- Consider implementing retention strategies or loyalty programs that reward customers for their longevity with the company.
- Explore options to optimize pricing strategies to make them more competitive in the market without compromising quality.
- Continuously collect feedback from customers who have churned to understand their reasons for leaving. Use this information to make improvements in areas that matter most to customer
