#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Required Libraries
import pandas as pd, numpy as np # For Data Manipulation
import statsmodels.api as sm, statsmodels.formula.api as smf # For Descriptive Statistics & Regression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder # For Encoding Categorical Data [Nominal | Ordinal]
from sklearn.preprocessing import OneHotEncoder # For Creating Dummy Variables of Categorical Data [Nominal]
from sklearn.impute import SimpleImputer, KNNImputer # For Imputation of Missing Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler # For Rescaling Data
from sklearn.model_selection import train_test_split # For Splitting Data into Training & Testing Sets
import pandas as pd, numpy as np # For Data Manipulation
import matplotlib.pyplot as plt, seaborn as sns # For Data Visualization
from sklearn.cluster import AgglomerativeClustering as agclus, KMeans as kmclus # For Agglomerative & K-Means Clustering
from sklearn.metrics import silhouette_score as sscore, davies_bouldin_score as dbscore # For Clustering Model Evaluation
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



# In[21]:


#Loading the first dataset
df1 = pd.read_excel(r"C:\Users\supri\Downloads\Applicant 1 CB Rejection Waterfall.xlsm", sheet_name='Sheet1')
df1


# In[22]:


print(df1)


# In[23]:


# Remove extra spaces from column names
df1.columns = df1.columns.str.strip()

# Verify the column names after removing extra spaces
print(df1.columns)


# In[24]:


df1.describe()


# # Data Preprocessing

# In[25]:


df1['DL Validation Mode Rule'] = df1['DL Validation Mode Rule'].fillna('No')
df1


# In[26]:


df1.isna().sum()


# In[27]:


df1.info()


# In[28]:


# Data Bifurcation
df_non_cat = df1[['Aadhaar And PAN Name Match Rule', 'Face Match Rule', 'Aadhaar And VID Name Match Rule', 'Age Rule',
       'Sixty Plus Dpd In Last 60 Months Ex Covid Period And Ex Gold Loan',
       'Inquiry In Last 6 Months Rule', '1 to 30 Dpd In Last 6 Months ExG',
       'Dpd Between 31 and 60 In Last 6 to 12 Months Ex Covid Period',
       'Written-off within threshold during COVID Period Rule',
       'Recent Credit Availed', 'Dpd between 31 and 60 In Last 6 Months',
       'Dpd Between 1 To 90 During Covid Period',
       'Written Off Rule In Whole Credit History', 'Mfi Accounts Rule',
       'bureauScore Rule', 'SMA Account in last 5 years Excluding COVID rule',
       'Ninety Plus DPD in gold Loan in 60 months Ex Covid rule',
       'Ninty Plus Dpd During Covid Rule',
       'Ninty Plus DPD in Last 60 Months Ex Gold and Ex Covid Period',
       '1 To 30 DPD In Last 24 to 12 Months Ex CP and ExG',
       'Total DPD for Gold Loans', 'Continous Credit History Rule',
       'Credit availed in last 24 months', 'Restructured Due to Covid Rule',
       'DPD more than 60 days in last 24 months',
       'DPD more than 60 days in 60 to 24 months',
       'DPD between 31 to 60 in Last 24 to 12 months Rule',
       'Gold SMA STD Accounts Rule',
       'Written-off beyond threshold during COVID Period Rule',
       'Total Gold Loan Overdue', 'Negative Accounts Rule',
       'Loan Marked Delinquent Rule',
       'Derogatory Account for Gold Loan Excluding COVID',
       'DPD more than 90 days in 60 to 24 months',
       'Current Overdue Amount Excluding Gold Loans Rule',
       '1 to 30 DPD in past 12 to 6 Months Rule',
       'Account Restructured And Closed Rule',
       'DPD more than 90 days in last 24 months',
       'Derogatory Accounts Ex Gold and Ex COVID Rule']] # Non-Categorical Data


# Remove currency sign and convert to float for non-categorical variables
for variable in df_non_cat:
    # Remove currency sign and commas, then convert to float
    df1[variable] = df1[variable].replace({'₹': '', ',': '', '%':''}, regex=True).astype(float)

# Convert non-currency columns to float
for col in df_non_cat:
    df_non_cat = df_non_cat.apply(pd.to_numeric, errors='coerce')

df_non_cat_float = df1[df_non_cat.columns]


# In[29]:


ki = KNNImputer(n_neighbors=5, weights='uniform') # weights : uniform | distance | {User Defined}
ki_fit = ki.fit_transform(df_non_cat_float) 
df_mdi_ki = pd.DataFrame(ki_fit, columns=df_non_cat_float.columns); df_mdi_ki # Missing Non-Categorical Data Imputed Subset using KNN Imputer
df_mdi_ki.info()


# In[30]:


# Optionally, drop the original non-categorical columns from df
df1.drop(columns=df_non_cat.columns, inplace=True)


# In[31]:


df1


# In[32]:


df1.info()


# In[33]:


columns_to_drop = ["Created At", "Aadhaar DL Name Match Rule", "VID Validation Mode Rule", "Pan Validation Mode Rule", "Aadhar Validation Mode Rule", "Aadhaar And Pan Linkage Rule", "DL Validation Mode Rule", "Fraud Rule"]

df1.drop(columns=columns_to_drop, inplace=True)


# In[34]:


df1


# In[35]:


df_mdi_ki


# In[36]:


df = pd.merge(df1, df_mdi_ki, left_index=True, right_index=True)
df


# In[37]:


df.to_excel("Preprocessed.xlsx")


# In[38]:


df.info()


# # Target Variable

# In[39]:


# explore the unique values in Rejection Reason column
df['Rejection Reason'].value_counts(normalize = True)


# In[40]:


# create a new column based on the loan_status column that will be our target variable
df['good_bad'] = np.where(df.loc[:, 'Rejection Reason'].isin(['CB Score', 'NTC (<30 Months)', '90+ DPD in last 24 months', 
                                                         '1-30 in 7 to24 months', '60+ DPD in last 24 months', 'Writen off during Covid', 
                                                         '31-60 DPD in 7-24 months', 'DPD 1-30 in last 6 months', 'MFI Rule', 
                                                         'Woff Ever (>500)', 'Current OD', 'DPD 31-60 in last 6 months']), 0, 1)
# Drop the original 'loan_status' column
df.drop(columns = ['Rejection Reason'], inplace = True)


# In[41]:


df


# In[42]:


df.info()


# # Data Visualisation

# In[43]:


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Determine the number of columns and rows for the subplots
num_cols = 3
num_rows = (len(numeric_columns) + num_cols - 1) // num_cols  # Calculate rows needed

# Set the figure size larger for better visibility
plt.figure(figsize=(num_cols * 6, num_rows * 5))

palette = sns.color_palette("viridis", len(numeric_columns))

# Create the subplots
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# In[44]:


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Determine the number of columns and rows for the subplots
num_cols = 3
num_rows = (len(numeric_columns) + num_cols - 1) // num_cols  # Calculate rows needed

# Set the figure size larger for better visibility
plt.figure(figsize=(num_cols * 6, num_rows * 5))

palette = sns.color_palette("viridis", len(numeric_columns))

# Create the subplots
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(df[col])
    plt.title(f'Box plot of {col}', fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# In[45]:


plt.figure(figsize=(6, 6))
sns.countplot(x='good_bad', data=df)
plt.title('Distribution of good_bad')
plt.xlabel('Good or Bad')
plt.ylabel('Count')
plt.show()


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate counts of each category
count = df['good_bad'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)

# Add a title and legend
plt.title('Distribution of good_bad')
plt.legend(title='Good or Bad', loc='upper right')

# Display the chart
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[47]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Recent Credit Availed', y='bureauScore Rule', hue='good_bad')
plt.title('Relationship between Recent Credit Availed and bureauScore Rule')
plt.xlabel('Recent Credit Availed')
plt.ylabel('bureauScore Rule')
plt.show()


# In[48]:


selected_columns = [
    'Aadhaar And PAN Name Match Rule', 'Face Match Rule', 'Aadhaar And VID Name Match Rule',
    'Age Rule', 'Recent Credit Availed', 'bureauScore Rule', 'good_bad'
]

# Create the pair plot
plt.figure(figsize=(15, 10))  # Adjust the size for better visibility
pair_plot = sns.pairplot(df[selected_columns], hue='good_bad', diag_kind='kde', palette='viridis')

# Adjust titles and labels for better readability
pair_plot.fig.suptitle('Pair Plot of Selected Columns', y=1.02, fontsize=16)
plt.show()


# In[50]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sidebar title
st.sidebar.title('Select Options')

# Sidebar options
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
selected_columns = st.sidebar.multiselect('Select Columns', numeric_columns)

# Plot type selection
plot_types = ['Histogram', 'Box Plot', 'Pie Chart', 'Scatter Plot', 'Pair Plot']
selected_plot = st.sidebar.selectbox('Select Plot Type', plot_types)

# Plot the selected graph
if selected_plot == 'Histogram':
    for col in selected_columns:
        st.write(f'## Distribution of {col}')
        st.pyplot(sns.histplot(df[col], kde=True))
elif selected_plot == 'Box Plot':
    for col in selected_columns:
        st.write(f'## Box Plot of {col}')
        st.pyplot(sns.boxplot(df[col]))
elif selected_plot == 'Pie Chart':
    st.write('## Distribution of good_bad')
    count = df['good_bad'].value_counts()
    plt.pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of good_bad')
    plt.axis('equal')
    st.pyplot(plt)
elif selected_plot == 'Scatter Plot':
    st.write('## Relationship between Columns')
    col1, col2 = st.sidebar.multiselect('Select Two Columns', selected_columns)
    sns.scatterplot(data=df, x=col1, y=col2, hue='good_bad')
    plt.title(f'Relationship between {col1} and {col2}')
    st.pyplot(plt)
elif selected_plot == 'Pair Plot':
    st.write('## Pair Plot of Selected Columns')
    selected_columns.append('good_bad')
    pair_plot = sns.pairplot(df[selected_columns], hue='good_bad', diag_kind='kde', palette='viridis')
    pair_plot.fig.suptitle('Pair Plot of Selected Columns', y=1.02, fontsize=16)
    st.pyplot(pair_plot)

