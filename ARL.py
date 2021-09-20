import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Reading our excel file
df_ = pd.read_excel('VBO_BootCamp_Donem/4.Hafta/online_retail_II.xlsx',sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

# TASK-1 - Data Preprocessing
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

# Reducing data for a single country.
df_grm = df[df['Country']=='Germany']   # We reduced our data to progress faster
df_grm['Country'].unique()
df_grm.head()

# TASK-2 - Creating Invoice-Description matrix and association Rules
# Here, we will recommend our customers a new product. Therefore, we should find a mathematical way to handle the issue.
# Our main goal will be to create invoice-description matrix which displays existence and non-existence of any products in baskets.
# To sum up, we will create a binary matrix containing 0 and 1.

# Let's start placing 'Invoice' variable in rows and 'Description' variable in columns.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().\
            unstack(). \
            fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum(). \
            unstack(). \
            fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

# Here we created 2 optional matrix because Invoice-Description matrix's output seems badly.
# Accordingly, I will be proceeding with Invoice-StockCode matrix to have more elegant output.
grm_inv_sto_df = create_invoice_product_df(df_grm, id=True)
grm_inv_sto_df.head()

# Creating Association Rules:
# We will use apriori algoritm to indicate association rules
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(grm_inv_sto_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
# Output explanation
# antecendents: id of the first product or products
# consequents: id of the second product or products
# antecedent support: probability of first product being observed alone
# consequent support: probability of the second product being observed alone
# support: probability of 2 products being observed together
# confidence: probability of buying 'second' when 'first' is bought.
# lift: increase in probability of buying product 'second' when 'first' is bought.
# leverage:Leverage effect. Similar to lift but it tends to prioritize higher 'support' values therefore it is biased
# conviction: expected frequency (expectation) of 'first' without 'second'

# The problem. Which variable should I focus on?
# Well, if we're going to take a specific action, it makes sense to focus on the 'lift' variable.
# Because on this study, we're working on 'associations'.
rules.sort_values("lift", ascending=False).head(50) # Let's check first 50 one
# Lift is a solid variable to find out which product will be bought(probably) after buying 'first' product.

# TASK-3 - Finding the name of the product by StockCodes
# If I need to check product name I'll create a new function to check name of any product as below.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)
# Examples:
check_id(df_grm, 21987)  # 'PACK OF 6 SKULL PAPER CUPS'
check_id(df_grm, 23235)  # 'STORAGE TIN VINTAGE LEAF'
check_id(df_grm, 22747)  # "POPPY'S PLAYHOUSE BATHROOM"

# TASK-4 - Product Recommendation(by 'lift')
sorted_rules = rules.sort_values("lift", ascending=False)

# Example users to recommend: 21987, 23235, 22747

product_id = 21987
recommendation_list = []
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
           # recommendation_list.pop(product_id)
22747 in recommendation_list # 2 products recomended

# TASK-5 - Names of the recommended products
check_id(df, 21086) # 'SET/6 RED SPOTTY PAPER CUPS'
check_id(df, 21988) # 'PACK OF 6 SKULL PAPER PLATES'
