import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df1 = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\Cab_Data.csv')
df2 = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\City.csv')
df3 = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\Customer_ID.csv')
df4 = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\Transaction_ID.csv')
df5 = pd.read_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\holidays.csv')
df5['date'] = pd.to_datetime(df5['date'], format='%d/%m/%Y')
merged_df = pd.merge(df1, df2, on='City', how='inner')
merged_df1=pd.merge(merged_df,df4,on='Transaction ID',how='inner')
merged_df2=pd.merge(merged_df1,df3,on='Customer ID',how='inner')
merged_df2['date'] = pd.to_datetime(merged_df2['Date of Travel'], unit='D', origin='1899-12-30')
merged_df3=pd.merge(merged_df2,df5,on='date',how='inner')
merged_df3.drop('Date of Travel',axis=1,inplace=True)
merged_df3.to_csv('C:\\Users\\anuti\\OneDrive\\Desktop\\internship\\G2M_Cab\\finalcab_data.csv', index=False)
merged_df3.drop_duplicates()
#print(merged_df2)
merged_df3.dropna()
#print(merged_df2)
from scipy.stats import zscore
df = merged_df3.apply(pd.to_numeric, errors='coerce')
def find_outliers(column):
    z_scores = np.abs((column - column.mean()) / column.std())
    return z_scores > 3  # Threshold for outlier detection

# Iterate over each column and find outliers
for column in df.columns:
    outliers = find_outliers(df[column])
    if outliers.any():
        print(f"Outliers detected in column '{column}':")
        print(df[outliers])
# Display cleaned DataFrame and outliers
print("Cleaned DataFrame:")
print(merged_df3)
print("\nOutliers:")
print(outliers)
merged_df3['Profit']=merged_df3['Price Charged']-merged_df3['Cost of Trip']
#print(merged_df3)
merged_df3['year'] = merged_df3['date'].dt.year
# Group by year and company name, then calculate yearly profit
yearly_profit = merged_df3.groupby(['year', 'Company'])['Profit'].sum().reset_index()

# Visualize the results 
yearly_profit.pivot(index='year', columns='Company', values='Profit').plot(kind='bar', figsize=(10, 6))
plt.title('Yearly Profit Analysis by Company')
plt.xlabel('Year')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend(title='Company')
plt.show()
#displaying seasonal profit based on holidays
seasonal_profit = merged_df3.groupby(['holiday', 'Company'])['Profit'].sum().reset_index()
seasonal_profit.pivot(index='holiday', columns='Company', values='Profit').plot(kind='bar', figsize=(10, 6))
plt.title('Seasonal Profit Analysis by Company')
plt.xlabel('holiday')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend(title='Company')
plt.show()
#dividing the income into different class labels
income_bins = [0, 10000, 20000, 30000, float('inf')]  # Define the income bins
income_labels = ['Very Low','Low','Medium', 'High']  # Define the labels for each bin

# Create a new column with income classes
merged_df3['income_class'] = pd.cut(merged_df3['Income (USD/Month)'], bins=income_bins, labels=income_labels, right=False)
#displaying income based profit analysis
income_profit = merged_df3.groupby(['income_class', 'Company'])['Profit'].sum().reset_index()
income_profit.pivot(index='Company', columns='income_class', values='Profit').plot(kind='bar',figsize=(10, 6))
legend2=['0-10000 very low','10000-20000 low','20000-30000 medium','>30000 high']
plt.title('Income Profit Analysis by Company')
plt.xlabel('Income_class')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend(legend2,title='Income')
plt.show()
#displaying gender based profit analysis
gender_profit = merged_df3.groupby(['Gender', 'Company'])['Profit'].sum().reset_index()
gender_profit.pivot(index='Gender', columns='Company', values='Profit').plot(kind='bar', figsize=(10, 6))
plt.title('Gender Profit Analysis by Company')
plt.xlabel('Gender')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend(title='Company')
plt.show()
age_bins = [0, 25, 40, 60, 100]  # Define the income bins
age_labels = ['Youth','Middle-aged','Older', 'Senior']  # Define the labels for each bin
merged_df3['age_class'] = pd.cut(merged_df3['Age'], bins=age_bins, labels=age_labels, right=False)
#displaying age based profit analysis
gender_profit = merged_df3.groupby(['age_class', 'Company'])['Profit'].sum().reset_index()
gender_profit.pivot(index='Company', columns='age_class', values='Profit').plot(kind='bar', figsize=(10, 6))
plt.title('Age Profit Analysis by Company')
plt.xlabel('Age_class')
plt.ylabel('Profit')
plt.xticks(rotation=45)
legend4=['0-25 youth','25-40 midddle-aged','40-60 older','>60 senior']
plt.legend(legend4,title='Age category')
plt.show()
#displaying city based profit analysis
city_profit = merged_df3.groupby(['City', 'Company'])['Profit'].sum().reset_index()
city_profit.pivot(index='City', columns='Company', values='Profit').plot(kind='bar', figsize=(10, 6))
plt.title('City Profit Analysis by Company')
plt.xlabel('City')
plt.ylabel('Profit')
plt.xticks(rotation=45)
plt.legend(title='Company')
plt.show()

# Remove commas from the 'Population' column
merged_df3['Population'] = merged_df3['Population'].str.replace(',', '')

# Convert the 'Population' column to numeric values
merged_df3['Population'] = pd.to_numeric(merged_df3['Population'])
merged_df3['Users'] = merged_df3['Users'].str.replace(',', '')

merged_df3['Users'] = pd.to_numeric(merged_df3['Users'])
# Group by country and calculate the total cab user count and population
country_stats = merged_df3.groupby(['City','Company']).agg({'Population': 'mean', 'Users': 'sum'})

# Calculate the percentage of people using cabs in each country
country_stats['Percentage_Using_Cabs'] = (country_stats['Users'] / country_stats['Population']) * 100
country_stats.reset_index(inplace=True)
# Plotting
plt.figure(figsize=(10, 6))
for company in country_stats['Company'].unique():
    company_data = country_stats[country_stats['Company'] == company]
    plt.plot(company_data['City'], company_data['Percentage_Using_Cabs'], label=company)

plt.xlabel('City')
plt.ylabel('Percentage of Cab Users')
plt.title('Percentage of Cab Users within Each City by Company')
plt.legend(title='Company')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

