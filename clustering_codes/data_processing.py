# path/filename: data_processing.py

import pandas as pd

def remove_companies(nasdaq_data, companies_to_remove):
    # Removes specified companies from the NASDAQ dataset
    return nasdaq_data.drop(companies_to_remove, axis=1, level=0)

def categorize_companies_by_risk(Clusters, empty_columns):
    # Categorizes companies into clusters based on risk levels
    companies_cleaned = Clusters.index.difference(empty_columns)
    Clusters.set_index(companies_cleaned, inplace=True)

    Cluster_0, Cluster_1, Cluster_2 = [], [], []
    for i in range(len(Clusters)):
        if Clusters.iloc[i, 0] == 0:
            Cluster_0.append(Clusters.index[i])
        elif Clusters.iloc[i, 0] == 1:
            Cluster_1.append(Clusters.index[i])
        else:
            Cluster_2.append(Clusters.index[i])
    
    return Cluster_0, Cluster_1, Cluster_2

def categorize_companies_by_return(Clusters, empty_columns):
    # Categorizes companies into clusters based on returns
    companies_cleaned = df.columns.difference(empty_columns)
    Clusters.set_index(companies_cleaned, inplace=True)

    Cluster_0_2, Cluster_1_2, Cluster_2_2 = [], [], []
    for i in range(len(Clusters)):
        if Clusters.iloc[i, 0] == 0:
            Cluster_0_2.append(Clusters.index[i])
        elif Clusters.iloc[i, 0] == 1:
            Cluster_1_2.append(Clusters.index[i])
        else:
            Cluster_2_2.append(Clusters.index[i])
    
    return Cluster_0_2, Cluster_1_2, Cluster_2_2

def determine_risk(select_company, Cluster_0, Cluster_1, Cluster_2):
    # Determines the risk category of a selected company
    if select_company in Cluster_0:
        return 'High Risk'
    elif select_company in Cluster_1:
        return 'Medium Risk'
    elif select_company in Cluster_2:
        return 'Low Risk'
    else:
        return 'Company not found'

# Example usage
if __name__ == "__main__":
    nasdaq100_data = pd.DataFrame()  # Load data here
    companies_to_remove = ['ABNB', 'CEG', 'DASH', 'GEHC', 'GFS']

    nasdaq100_cleaned = remove_companies(nasdaq100_data, companies_to_remove)
    # Assuming Clusters and empty_columns are defined
    Cluster_0, Cluster_1, Cluster_2 = categorize_companies_by_risk(Clusters, empty_columns)
    Cluster_0_2, Cluster_1_2, Cluster_2_2 = categorize_companies_by_return(Clusters, empty_columns)

    select_company = 'AAPL'
    risk_category = determine_risk(select_company, Cluster_0, Cluster_1, Cluster_2)
    print(risk_category)
