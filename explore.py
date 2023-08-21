import pandas as pd

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
