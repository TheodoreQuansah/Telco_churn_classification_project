import pandas as pd

from env import get_connection

################################################################# acquire main function#################################################################

def get_telco_data():
    
    db_url = get_connection('telco_churn')

    query = '''
           SELECT 
        customers.customer_id,
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

     # Convert "total_charges" to numeric (this will handle any non-numeric values)
    telco_df['total_charges'] = pd.to_numeric(telco_df['total_charges'], errors='coerce')
    # Replace any NaN values with 0 (or any other suitable value)
    telco_df['total_charges'].fillna(0, inplace=True)
    # Convert "total_charges" to integer
    telco_df['total_charges'] = telco_df['total_charges'].astype(int)
    
    return telco_df


#################################### encoding and cleaning data columns #######################################################################

def encode_columns(train, val, test):
    # One-hot encoding categorical columns with get_dummies
    train = pd.get_dummies(train, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)
    
    val = pd.get_dummies(val, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)
    
    test = pd.get_dummies(test, columns=[
        'gender', 'senior_citizen', 'partner', 'dependents', 'phone_service',
        'multiple_lines', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'payment_type',
        'contract_type', 'internet_service_type', 'paperless_billing'
    ], drop_first=True)

    # Drop columns with 'No internet service' and 'internet_service_type_None'
    cols_to_drop = train.columns[train.columns.str.contains('No internet service')]
    train = train.drop(columns=cols_to_drop)
    train = train.drop(columns=['internet_service_type_None'])
    train = train.rename(columns={'gender_male': 'gender'})

    cols_to_drop = val.columns[val.columns.str.contains('No internet service')]
    val = val.drop(columns=cols_to_drop)
    val = val.drop(columns=['internet_service_type_None'])
    val = val.rename(columns={'gender_male': 'gender'})

    cols_to_drop = test.columns[test.columns.str.contains('No internet service')]
    test = test.drop(columns=cols_to_drop)
    test = test.drop(columns=['internet_service_type_None'])
    test = test.rename(columns={'gender_male': 'gender'})

    # Rename columns by removing "_Yes"
    columns_to_rename = [
        "partner_Yes", "dependents_Yes", "phone_service_Yes", 
        "multiple_lines_Yes", "online_security_Yes", "online_backup_Yes", 
        "device_protection_Yes", "tech_support_Yes", "streaming_tv_Yes", 
        "streaming_movies_Yes", "paperless_billing_Yes"
    ]
    for col in columns_to_rename:
        new_name = col.replace("_Yes", "")
        train.rename(columns={col: new_name}, inplace=True)
        val.rename(columns={col: new_name}, inplace=True)
        test.rename(columns={col: new_name}, inplace=True)

    # Rename columns based on provided replacements
    columns_to_rename = [
        "senior_citizen_1", "payment_type_Manual Payment", 
        "contract_type_one/two-years", "internet_service_type_Fiber optic"
    ]
    for col in columns_to_rename:
        new_name = col.replace("senior_citizen_1", "senior_citizen").replace("payment_type_Manual Payment",
                                                                             "payment_type").replace("contract_type_one/two-years", 
                                                                             "contract_type").replace("internet_service_type_Fiber optic", 
                                                                             "internet_service_type")
        train.rename(columns={col: new_name}, inplace=True)
        val.rename(columns={col: new_name}, inplace=True)
        test.rename(columns={col: new_name}, inplace=True)
    
    return train, val, test
