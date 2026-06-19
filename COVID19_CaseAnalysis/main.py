#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv("data/patient.csv")

#dispaly first records
df.head()

#check columns
df.columns

#check shape
df.shape

#check datatypes
df.info()

#data cleaning
df.isnull().sum() #check duplicate values
df.drop_duplicates(inplace=True)#remove duplicate values

#convert data columns
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'])
df['released_date'] = pd.to_datetime(df['released_date'])
df['deceased_date'] = pd.to_datetime(df['deceased_date'])

#create age column
df['age'] = 2020 - df['birth_year']
df[['birth_year','age']].head()
print(df.isnull().sum())

#convert data columns
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'])

df['released_date'] = pd.to_datetime(df['released_date'])

df['deceased_date'] = pd.to_datetime(df['deceased_date'])
df.info()

#Recovery duration
df['recovery_days'] = (
    df['released_date']
    - df['confirmed_date']
).dt.days
print(df[['confirmed_date',
          'released_date',
          'recovery_days']].head())
print(df['recovery_days'].describe())

#EDA
#gender analysis
print(df['sex'].value_counts())
sns.countplot(x='sex', data=df)
plt.title('Gender Distribution')
plt.show()


#analyze patient outcomes
print(df['state'].value_counts())
sns.countplot(x='state', data=df)
plt.title('Patient Outcomes')
plt.show()

#recovery histogram
sns.histplot(df['recovery_days'].dropna(), bins=10)
plt.title('Recovery Days Distribution')
plt.xlabel('Recovery Days')
plt.ylabel('Number of Patients')
plt.show()
print(df['sex'].value_counts())
print(df['state'].value_counts())
sns.countplot(x='state', data=df)
plt.title("Patient Outcomes")
plt.show()

#country analysis
print(df['country'].value_counts())

#region analysis
print(df['region'].value_counts().head(10))

#analyze infection sources
print(df['infection_reason'].value_counts())

print(df['country'].value_counts())
print(df['region'].value_counts().head(10))
print(df['infection_reason'].value_counts())

#correlation analysis
recovery_df = df[df['recovery_days'].notna()]

corr = recovery_df[
    ['age',
     'contact_number',
     'infection_order',
     'recovery_days']
].corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()

#Linear regression
from sklearn.linear_model import LinearRegression

recovery_df = df[df['recovery_days'].notna()]

X = recovery_df[['age',
                 'contact_number',
                 'infection_order']]

X = X.fillna(X.mean())

y = recovery_df['recovery_days']

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
