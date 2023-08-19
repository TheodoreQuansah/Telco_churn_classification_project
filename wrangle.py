import pandas as pd

from scipy import stats
from env import get_connection
from sklearn.model_selection import train_test_split

################################################################# acquire main function#################################################################

def get_telco_data():
    
    db_url = get_connection('telco_churn')

    query = '''
           SELECT 
        customers.gender,
        customers.senior_citizen,
        customers.partner,
        customers.dependents,
        customers.tenure,
        customers.phone_service,
        customers.multiple_lines,
        customers.online_security,
        customers.online_backup,
        customers.device_protection,
        customers.tech_support,
        customers.streaming_tv,
        customers.streaming_movies,
        customers.paperless_billing,
        customers.monthly_charges,
        customers.total_charges,
        customers.churn,
        payment_types.payment_type,
        contract_types.contract_type,
        internet_service_types.internet_service_type
    FROM
        customers
            LEFT JOIN
        customer_details ON customer_details.customer_id = customers.customer_id
            LEFT JOIN
        customer_contracts ON customer_contracts.customer_id = customer_details.customer_id
            LEFT JOIN
        customer_payments ON customer_payments.customer_id = customer_contracts.customer_id
            LEFT JOIN
        customer_signups ON customer_signups.customer_id = customer_payments.customer_id
            LEFT JOIN
        customer_subscriptions ON customer_subscriptions.customer_id = customer_signups.customer_id
            LEFT JOIN
        customer_churn ON customer_churn.customer_id = customer_subscriptions.customer_id
            LEFT JOIN
        payment_types ON payment_types.payment_type_id = customers.payment_type_id
            LEFT JOIN
        contract_types ON contract_types.contract_type_id = customer_contracts.contract_type_id
            LEFT JOIN
        internet_service_types ON internet_service_types.internet_service_type_id = customers.internet_service_type_id;
            '''

    #read sql query into a dataframe
    telco_df = pd.read_sql(query, db_url)

    #replace all the total charges rows with no values with 0
    telco_df['total_charges'] = telco_df['total_charges'].replace(' ', 0)

    #replacing all the no internet service to No
    telco_df['multiple_lines'] = telco_df['multiple_lines'].replace('No phone service', 'No')
    telco_df['online_security'] = telco_df['online_security'].replace('No phone service', 'No')
    telco_df['online_backup'] = telco_df['online_backup'].replace('No phone service', 'No')
    telco_df['device_protection'] = telco_df['device_protection'].replace('No phone service', 'No')
    telco_df['tech_support'] = telco_df['tech_support'].replace('No phone service', 'No')
    telco_df['streaming_movies'] = telco_df['streaming_movies'].replace('No phone service', 'No')
    telco_df['streaming_tv'] = telco_df['streaming_tv'].replace('No phone service', 'No')

    #grouping all the value counts in payment type into manual and electronic payments
    telco_df.loc[telco_df['payment_type'].str.contains('automatic', case=False), 'payment_type'] = 'Automatic Payment'
    telco_df.loc[telco_df['payment_type'].str.contains('check', case=False), 'payment_type'] = 'Manual Payment'

    #grouping all the value counts in contract type into monthly and yearly contracts
    telco_df.loc[telco_df['contract_type'].str.contains('year', case=False), 'contract_type'] = 'one/two-years'

    #changing all the values in churn to boolean
    telco_df['churn'] = telco_df['churn'].replace('Yes', True)
    telco_df['churn'] = telco_df['churn'].replace('No', False)
    
    return telco_df








#################################### train, validate and test split  #######################################################################

def train_val_test(df, strat, seed=42):
    
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
    
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
    
    return train, val, test






#################################### Chi^2 test #######################################################################
def perform_chi_squared_test(df, feature1, feature2, alpha=0.05):
    ct = pd.crosstab(df[feature1], df[feature2])
    chi, p, degf, exp = stats.chi2_contingency(ct)

    if p < alpha:
        result = 'we reject the null hypothesis. There appears to be a relationship'
    else:
        result = 'we fail to reject the null hypothesis'

    return result, p





#################################### t-test #######################################################################
def perform_t_test(df, feature, alpha=0.05):
    tenure_churn_yes = df[df['churn'] == True][feature]
    tenure_churn_no = df[df['churn'] == False][feature]

    t, p = stats.ttest_ind(tenure_churn_yes, tenure_churn_no, equal_var=False)

    if p < alpha:
        result = "Reject the null hypothesis: {} has a significant effect on churn.".format(feature)
    else:
        result = "Fail to reject the null hypothesis: {} does not have a significant effect on churn.".format(feature)

    return result, p




