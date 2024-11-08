from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('../data/Online_Retail.csv', encoding='unicode_escape')

df.dropna(subset=['Description', 'InvoiceNo'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]  # Remove rows with negative quantities
df['InvoiceNo'] = df['InvoiceNo'].astype(str)

basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    return 1 if x >= 1 else 0

basket = basket.map(encode_units)

min_supp = 0.01
min_thresh = 2

frequent_itemsets_fpgrowth = fpgrowth(basket, min_support=min_supp, use_colnames=True)
frequent_itemsets_fpgrowth = frequent_itemsets_fpgrowth.sort_values(by='support', ascending=False)

fpgrowth_rules = association_rules(frequent_itemsets_fpgrowth, metric='lift', min_threshold=min_thresh)

fpgrowth_rules['antecedents'] = fpgrowth_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
fpgrowth_rules['consequents'] = fpgrowth_rules['consequents'].apply(lambda x: ', '.join(list(x)))

frequent_itemsets_fpgrowth.to_csv('../data/frequent_itemsets_fpgrowth.csv', index=False)
fpgrowth_rules.to_csv('../data/rules_fpgrowth.csv', index=False)

print(frequent_itemsets_fpgrowth.head())
print(fpgrowth_rules.head())