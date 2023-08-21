import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

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


#################################### churn plot #######################################################################

def create_churn_plots(train):
    # Create subplots with two plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Countplot for Senior Citizens
    sns.countplot(data=train, x='senior_citizen', hue='churn', ax=axes[0])
    axes[0].set_xlabel('Senior Citizen')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Churn Distribution by Senior Citizen Status')

    # Plot 2: Pie Chart for Senior Citizens
    # Calculate churn rates
    churn_rates = train.groupby('senior_citizen')['churn'].mean() * 100

    # Create a bar plot
    churn_rates.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.xlabel('Senior Citizen')
    plt.ylabel('Churn Rate (%)')
    plt.title('Churn Rate by Senior Citizen Status')
    plt.xticks(ticks=[0, 1], labels=['Not Senior', 'Senior'], rotation=0)
    plt.ylim(0, 50)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the subplots
    plt.show()


#################################### tech_support vs churn plot #######################################################################

def create_tech_support_churn_plot(data):
    # Set the Seaborn style to "whitegrid" for the plot
    sns.set(style="whitegrid")

    # Create a figure with a specific size (8 units wide, 6 units tall)
    plt.figure(figsize=(8, 6))

    # Create a countplot using Seaborn to visualize the distribution of 'tech_support' by 'churn'
    sns.countplot(data=data, x='tech_support', hue='churn')

    # Set the title of the plot
    plt.title("Tech support Distribution by Churn")

    # Label the x-axis with more descriptive labels
    plt.xlabel("Tech support (No = No tech support, Yes = Has tech support)")

    # Label the y-axis
    plt.ylabel("Count")

    # Display the plot
    plt.show()


#################################### tenure vs churn plot #######################################################################

def create_tenure_churn_boxplot(data):
    # Create a figure with a specific size (8 units wide, 5 units tall)
    plt.figure(figsize=(8, 5))

    # Create a boxplot using Seaborn to visualize the distribution of 'tenure' by 'churn'
    # Customize colors based on 'churn' values (False: turquoise, True: orange)
    sns.boxplot(data=data, x='churn', y='tenure', palette={False: 'turquoise', True: 'orange'})

    # Set the title of the plot
    plt.title('Average Tenure by Churn')

    # Label the x-axis as 'Churn'
    plt.xlabel('Churn')

    # Label the y-axis as 'Average Tenure'
    plt.ylabel('Average Tenure')

    # Add a legend with custom labels ("Not Churned" and "Churned")
    plt.legend(title='Churn', labels=["Not Churned", "Churned"])

    # Display the plot
    plt.show()


#################################### contract_type vs churn plot #######################################################################

def create_contract_type_churn_plot(data):
    # Set the Seaborn style to "whitegrid" for the plot
    sns.set(style="whitegrid")

    # Create a figure with a specific size (8 units wide, 6 units tall)
    plt.figure(figsize=(8, 6))

    # Create a countplot using Seaborn to visualize the distribution of 'contract_type' by 'churn'
    sns.countplot(data=data, x='contract_type', hue='churn')

    # Set the title of the plot
    plt.title("Distribution of Contract Types by Churn")

    # Label the x-axis as 'Contract Type'
    plt.xlabel("Contract Type")

    # Label the y-axis as 'Count'
    plt.ylabel("Count")

    # Display the plot
    plt.show()


#################################### internet_service_type vs churn plot #######################################################################

def create_internet_service_churn_plot(data):
    # Create a contingency table (crosstab) of 'internet_service_type' and 'churn' in the data
    ct = pd.crosstab(data['internet_service_type'], data['churn'])

    # Create a stacked bar chart to visualize the relationship between 'Internet Service Type' and 'Churn'
    ct.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Label the x-axis as 'Internet Service Type'
    plt.xlabel('Internet Service Type')

    # Label the y-axis as 'Count'
    plt.ylabel('Count')

    # Set the title of the plot
    plt.title('Relationship between Internet Service Type and Churn')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend with custom labels ('No' and 'Yes' for 'Churn')
    plt.legend(title='Churn', labels=['No', 'Yes'])

    # Add grid lines to the plot
    plt.grid(True)

    # Display the plot
    plt.show()


