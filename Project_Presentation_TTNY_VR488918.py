# pandas,numpy library
import pandas as pd
import numpy as np
# matplotlib,Charts, the model, Binning and Report library
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess
from optbinning import OptimalBinning
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pickle

st.markdown("<h1 style='text-align: center; color: grey;'>PROGRAMING FOR DATA SCIENCE</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>The Final Project With Finance Data </h2>", unsafe_allow_html=True)
st.markdown('**Annt**: Niccolò Marastoni')
st.markdown('**Prepared by**: Thi Nhu Y Tran - VR488918')
st.header(':blue[Introduction]')
st.write('Home Credit is a big financial company in the world, \
they pthey published their customer data in Kaggle to find the best solution for the question:\
          Can you predict how capable each applicant is of repaying a loan? from Data Scientists.')
st.write('In the project, we discover the structure, features,\
          and meaning of each feature of the data relative to the financial industry through Python programming. In addition,\
          the tools, and library support our to explore the customer portrait of HomeCredit.')
st.write('Cleaning the data is the most important task, it helps the person who works in the analysis industry have the correct analysis and insight about information from data. \
         Therefore, we also show the step-by-step cleaning data by using the tools in Python.')
st.write('In the last part of the project, we use Weight of Evidence (WOE) and Information value (IV) indicator to evaluate the impact of features (variables)\
          on to target variable and use Logistic regression to  predict Good/ Bad customers.')
# st.write('In the final part of project, we continue go to some models in Machine learning.  ')
st.subheader('I. Data Exploration and Data Wrangeling')

st.write ('We explore two data files including "**application_train.csv**" and "**application_test.csv**" \
         in the following link: https://www.kaggle.com/competitions/home-credit-default-risk/data')
st.write ('After researching and cleaning two files, we will use "**Train data**" to build the models, \
          and using "**Test data**" as **Out of Time** data to test the models.')

st.subheader('1. The data "application_train.csv".')
st.write('**a. Explore the datasets**')

#  Read "application_train.csv" from a folder on the local computer
application_train = pd.read_csv(r'C:\Users\YTRAN\OneDrive - Università degli Studi di Verona\Programing and Database\Project\Data\application_train.csv')
st.write('We have the overview the dataset as :blue[**Table 1**].')
st.write('_:blue[**Table 1**]: **The structure of "application_train.csv"**_')
st.write(application_train.head(10))
st.write('The dataset has **307511 rows** and **122 features (columns)**.')


st.write('_:blue[**Table 2**]: **The information of the first 60 features.**_')
#st.write('### Test Markdown')
application_train_0_60 = application_train.iloc[:,0:60].info()
application_train_from_60 = application_train.iloc[:,60:].info()

# The information of the first 60 features
st.text("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 307511 entries, 0 to 307510
Data columns (total 60 columns):
 #   Column                       Non-Null Count   Dtype  
---  ------                       --------------   -----  
 0   SK_ID_CURR                   307511 non-null  int64  
 1   TARGET                       307511 non-null  int64  
 2   NAME_CONTRACT_TYPE           307511 non-null  object 
 3   CODE_GENDER                  307511 non-null  object 
 4   FLAG_OWN_CAR                 307511 non-null  object 
 5   FLAG_OWN_REALTY              307511 non-null  object 
 6   CNT_CHILDREN                 307511 non-null  int64  
 7   AMT_INCOME_TOTAL             307511 non-null  float64
 8   AMT_CREDIT                   307511 non-null  float64
 9   AMT_ANNUITY                  307499 non-null  float64
 10  AMT_GOODS_PRICE              307233 non-null  float64
 11  NAME_TYPE_SUITE              306219 non-null  object 
 12  NAME_INCOME_TYPE             307511 non-null  object 
 13  NAME_EDUCATION_TYPE          307511 non-null  object 
 14  NAME_FAMILY_STATUS           307511 non-null  object 
 15  NAME_HOUSING_TYPE            307511 non-null  object 
 16  REGION_POPULATION_RELATIVE   307511 non-null  float64
 17  DAYS_BIRTH                   307511 non-null  int64  
 18  DAYS_EMPLOYED                307511 non-null  int64  
 19  DAYS_REGISTRATION            307511 non-null  float64
 20  DAYS_ID_PUBLISH              307511 non-null  int64  
 21  OWN_CAR_AGE                  104582 non-null  float64
 22  FLAG_MOBIL                   307511 non-null  int64  
 23  FLAG_EMP_PHONE               307511 non-null  int64  
 24  FLAG_WORK_PHONE              307511 non-null  int64  
 25  FLAG_CONT_MOBILE             307511 non-null  int64  
 26  FLAG_PHONE                   307511 non-null  int64  
 27  FLAG_EMAIL                   307511 non-null  int64  
 28  OCCUPATION_TYPE              211120 non-null  object 
 29  CNT_FAM_MEMBERS              307509 non-null  float64
 30  REGION_RATING_CLIENT         307511 non-null  int64  
 31  REGION_RATING_CLIENT_W_CITY  307511 non-null  int64  
 32  WEEKDAY_APPR_PROCESS_START   307511 non-null  object 
 33  HOUR_APPR_PROCESS_START      307511 non-null  int64  
 34  REG_REGION_NOT_LIVE_REGION   307511 non-null  int64  
 35  REG_REGION_NOT_WORK_REGION   307511 non-null  int64  
 36  LIVE_REGION_NOT_WORK_REGION  307511 non-null  int64  
 37  REG_CITY_NOT_LIVE_CITY       307511 non-null  int64  
 38  REG_CITY_NOT_WORK_CITY       307511 non-null  int64  
 39  LIVE_CITY_NOT_WORK_CITY      307511 non-null  int64  
 40  ORGANIZATION_TYPE            307511 non-null  object 
 41  EXT_SOURCE_1                 134133 non-null  float64
 42  EXT_SOURCE_2                 306851 non-null  float64
 43  EXT_SOURCE_3                 246546 non-null  float64
 44  APARTMENTS_AVG               151450 non-null  float64
 45  BASEMENTAREA_AVG             127568 non-null  float64
 46  YEARS_BEGINEXPLUATATION_AVG  157504 non-null  float64
 47  YEARS_BUILD_AVG              103023 non-null  float64
 48  COMMONAREA_AVG               92646 non-null   float64
 49  ELEVATORS_AVG                143620 non-null  float64
 50  ENTRANCES_AVG                152683 non-null  float64
 51  FLOORSMAX_AVG                154491 non-null  float64
 52  FLOORSMIN_AVG                98869 non-null   float64
 53  LANDAREA_AVG                 124921 non-null  float64
 54  LIVINGAPARTMENTS_AVG         97312 non-null   float64
 55  LIVINGAREA_AVG               153161 non-null  float64
 56  NONLIVINGAPARTMENTS_AVG      93997 non-null   float64
 57  NONLIVINGAREA_AVG            137829 non-null  float64
 58  APARTMENTS_MODE              151450 non-null  float64
 59  BASEMENTAREA_MODE            127568 non-null  float64
dtypes: float64(27), int64(21), object(12)
memory usage: 140.8+ MB
""")

# Information of the remaing features
st.write('_:blue[**Table 3**]: **The information of the remaining features.**_')
st.text("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 307511 entries, 0 to 307510
Data columns (total 62 columns):
 #   Column                        Non-Null Count   Dtype  
---  ------                        --------------   -----  
 0   YEARS_BEGINEXPLUATATION_MODE  157504 non-null  float64
 1   YEARS_BUILD_MODE              103023 non-null  float64
 2   COMMONAREA_MODE               92646 non-null   float64
 3   ELEVATORS_MODE                143620 non-null  float64
 4   ENTRANCES_MODE                152683 non-null  float64
 5   FLOORSMAX_MODE                154491 non-null  float64
 6   FLOORSMIN_MODE                98869 non-null   float64
 7   LANDAREA_MODE                 124921 non-null  float64
 8   LIVINGAPARTMENTS_MODE         97312 non-null   float64
 9   LIVINGAREA_MODE               153161 non-null  float64
 10  NONLIVINGAPARTMENTS_MODE      93997 non-null   float64
 11  NONLIVINGAREA_MODE            137829 non-null  float64
 12  APARTMENTS_MEDI               151450 non-null  float64
 13  BASEMENTAREA_MEDI             127568 non-null  float64
 14  YEARS_BEGINEXPLUATATION_MEDI  157504 non-null  float64
 15  YEARS_BUILD_MEDI              103023 non-null  float64
 16  COMMONAREA_MEDI               92646 non-null   float64
 17  ELEVATORS_MEDI                143620 non-null  float64
 18  ENTRANCES_MEDI                152683 non-null  float64
 19  FLOORSMAX_MEDI                154491 non-null  float64
 20  FLOORSMIN_MEDI                98869 non-null   float64
 21  LANDAREA_MEDI                 124921 non-null  float64
 22  LIVINGAPARTMENTS_MEDI         97312 non-null   float64
 23  LIVINGAREA_MEDI               153161 non-null  float64
 24  NONLIVINGAPARTMENTS_MEDI      93997 non-null   float64
 25  NONLIVINGAREA_MEDI            137829 non-null  float64
 26  FONDKAPREMONT_MODE            97216 non-null   object 
 27  HOUSETYPE_MODE                153214 non-null  object 
 28  TOTALAREA_MODE                159080 non-null  float64
 29  WALLSMATERIAL_MODE            151170 non-null  object 
 30  EMERGENCYSTATE_MODE           161756 non-null  object 
 31  OBS_30_CNT_SOCIAL_CIRCLE      306490 non-null  float64
 32  DEF_30_CNT_SOCIAL_CIRCLE      306490 non-null  float64
 33  OBS_60_CNT_SOCIAL_CIRCLE      306490 non-null  float64
 34  DEF_60_CNT_SOCIAL_CIRCLE      306490 non-null  float64
 35  DAYS_LAST_PHONE_CHANGE        307510 non-null  float64
 36  FLAG_DOCUMENT_2               307511 non-null  int64  
 37  FLAG_DOCUMENT_3               307511 non-null  int64  
 38  FLAG_DOCUMENT_4               307511 non-null  int64  
 39  FLAG_DOCUMENT_5               307511 non-null  int64  
 40  FLAG_DOCUMENT_6               307511 non-null  int64  
 41  FLAG_DOCUMENT_7               307511 non-null  int64  
 42  FLAG_DOCUMENT_8               307511 non-null  int64  
 43  FLAG_DOCUMENT_9               307511 non-null  int64  
 44  FLAG_DOCUMENT_10              307511 non-null  int64  
 45  FLAG_DOCUMENT_11              307511 non-null  int64  
 46  FLAG_DOCUMENT_12              307511 non-null  int64  
 47  FLAG_DOCUMENT_13              307511 non-null  int64  
 48  FLAG_DOCUMENT_14              307511 non-null  int64  
 49  FLAG_DOCUMENT_15              307511 non-null  int64  
 50  FLAG_DOCUMENT_16              307511 non-null  int64  
 51  FLAG_DOCUMENT_17              307511 non-null  int64  
 52  FLAG_DOCUMENT_18              307511 non-null  int64  
 53  FLAG_DOCUMENT_19              307511 non-null  int64  
 54  FLAG_DOCUMENT_20              307511 non-null  int64  
 55  FLAG_DOCUMENT_21              307511 non-null  int64  
 56  AMT_REQ_CREDIT_BUREAU_HOUR    265992 non-null  float64
 57  AMT_REQ_CREDIT_BUREAU_DAY     265992 non-null  float64
 58  AMT_REQ_CREDIT_BUREAU_WEEK    265992 non-null  float64
 59  AMT_REQ_CREDIT_BUREAU_MON     265992 non-null  float64
 60  AMT_REQ_CREDIT_BUREAU_QRT     265992 non-null  float64
 61  AMT_REQ_CREDIT_BUREAU_YEAR    265992 non-null  float64
dtypes: float64(38), int64(20), object(4)
memory usage: 145.5+ MB
""")

st.warning(" Based on :blue[**Table 2**] and :blue[**Table 3**], there are a lot of null values in data. In the finance industry, \
           some reasons lead to null data such as: The customers has no information about some features, system errors in the \
           data collection.")
st.write('_:blue[**Table 4**]: **The statistical information of the features.**_')
st.write(application_train.describe().T)
st.write('In :blue[**Table 4**], we are able to the useful information as Maximum, Minimum, Mean, and Standard Deviation of each feature. \
            In some situations, these values represent the instability of the data. With **"application_train.csv"**, there are some time features have negative values including:\
          **DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE**.')
st.write('The cause of the negative values is these features that were calculated by the following formulas:')

st.info(f"""
* Client's age in days at the time of application: 
**DAYS_BIRTH = BIRTH DATE - APPLICATION DATE**
* How many days before the application the person started current employment: 
**DAYS_EMPLOYED = EMPLOYED DATE - APPLICATION DATE**\n
Similarities for the remaining variables such as: **DAYS_REGISTRATION, DAYS_ID_PUBLISH, DAYS_LAST_PHONE_CHANGE**
""")
DAYS_EMPLOYED_max = application_train['DAYS_EMPLOYED'].max()
st.write('In particular, the maximum value of **DAYS_EMPLOYED** is',DAYS_EMPLOYED_max, 'days ~ 1000 years, which is an outlier and impossible value.\
          Therefore, we replace it with **NaN** in the cleaning step.')

# b. Clean up the dataset
st.write('**b. Clean up the dataset**')
st.write('Following the above analysis of **DAYS_EMPLOYED**. We will replace', DAYS_EMPLOYED_max, 'by NaN such as:')

# Replace '365243' in  DAYS_EMPLOYED by NaN
application_train['DAYS_EMPLOYED'].replace(365243,np.nan,inplace = True)
st.code('''application_train['DAYS_EMPLOYED'].replace(365243,np.nan,inplace = True)''',language= 'python')

# Caculate the percentage of null value of each features.
st.write('In the next step, we calcualte the percentage of null values of each features.')
count_nullvalue = application_train.isna().sum(axis = 0).sort_values(ascending = False) 
per_nullvalue = count_nullvalue/application_train.shape[0]

# The table of percentage of null value
per_nullvalue_DF = pd.DataFrame(per_nullvalue,columns=['Per_NullValue'])
st.write('_:blue[**Table 5**]: **The percentage of null values of each features.**_')
st.write(per_nullvalue_DF)

st.warning('In :blue[**Table 5**], with the columns have the null values percentage is :red[larger than 40%], \
           when using them in order to build the\
         models, it will be hard to predict with the high exact level.\
            Therefore, we drop these features from the dataset. The following syntax is to remove these features:')

# The amount of variales that have the null value percentage > 40%
per_nullvalue_greater40 = per_nullvalue[per_nullvalue >= 0.4]
# The droping of variales that have the null value percentage > 40% 
cleanup_application_train = application_train.drop(columns = [i for i in per_nullvalue_greater40.index])
cleanup_application_train.info()

st.code('''
per_nullvalue_greater40 = per_nullvalue[per_nullvalue >= 0.4]
application_train.drop(columns = [i for i in per_nullvalue_greater40.index])
''')
# After removing the features have the per_nullvalue >= 0.4
st.write('_:blue[**Table 6**]: **The information of the remaining features after removing the features that have \
         the null values percentage >= 40%.**_')
st.text("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 307511 entries, 0 to 307510
Data columns (total 73 columns):
 #   Column                       Non-Null Count   Dtype  
---  ------                       --------------   -----  
 0   SK_ID_CURR                   307511 non-null  int64  
 1   TARGET                       307511 non-null  int64  
 2   NAME_CONTRACT_TYPE           307511 non-null  object 
 3   CODE_GENDER                  307511 non-null  object 
 4   FLAG_OWN_CAR                 307511 non-null  object 
 5   FLAG_OWN_REALTY              307511 non-null  object 
 6   CNT_CHILDREN                 307511 non-null  int64  
 7   AMT_INCOME_TOTAL             307511 non-null  float64
 8   AMT_CREDIT                   307511 non-null  float64
 9   AMT_ANNUITY                  307499 non-null  float64
 10  AMT_GOODS_PRICE              307233 non-null  float64
 11  NAME_TYPE_SUITE              306219 non-null  object 
 12  NAME_INCOME_TYPE             307511 non-null  object 
 13  NAME_EDUCATION_TYPE          307511 non-null  object 
 14  NAME_FAMILY_STATUS           307511 non-null  object 
 15  NAME_HOUSING_TYPE            307511 non-null  object 
 16  REGION_POPULATION_RELATIVE   307511 non-null  float64
 17  DAYS_BIRTH                   307511 non-null  int64  
 18  DAYS_EMPLOYED                252137 non-null  float64
 19  DAYS_REGISTRATION            307511 non-null  float64
 20  DAYS_ID_PUBLISH              307511 non-null  int64  
 21  FLAG_MOBIL                   307511 non-null  int64  
 22  FLAG_EMP_PHONE               307511 non-null  int64  
 23  FLAG_WORK_PHONE              307511 non-null  int64  
 24  FLAG_CONT_MOBILE             307511 non-null  int64  
 25  FLAG_PHONE                   307511 non-null  int64  
 26  FLAG_EMAIL                   307511 non-null  int64  
 27  OCCUPATION_TYPE              211120 non-null  object 
 28  CNT_FAM_MEMBERS              307509 non-null  float64
 29  REGION_RATING_CLIENT         307511 non-null  int64  
 30  REGION_RATING_CLIENT_W_CITY  307511 non-null  int64  
 31  WEEKDAY_APPR_PROCESS_START   307511 non-null  object 
 32  HOUR_APPR_PROCESS_START      307511 non-null  int64  
 33  REG_REGION_NOT_LIVE_REGION   307511 non-null  int64  
 34  REG_REGION_NOT_WORK_REGION   307511 non-null  int64  
 35  LIVE_REGION_NOT_WORK_REGION  307511 non-null  int64  
 36  REG_CITY_NOT_LIVE_CITY       307511 non-null  int64  
 37  REG_CITY_NOT_WORK_CITY       307511 non-null  int64  
 38  LIVE_CITY_NOT_WORK_CITY      307511 non-null  int64  
 39  ORGANIZATION_TYPE            307511 non-null  object 
 40  EXT_SOURCE_2                 306851 non-null  float64
 41  EXT_SOURCE_3                 246546 non-null  float64
 42  OBS_30_CNT_SOCIAL_CIRCLE     306490 non-null  float64
 43  DEF_30_CNT_SOCIAL_CIRCLE     306490 non-null  float64
 44  OBS_60_CNT_SOCIAL_CIRCLE     306490 non-null  float64
 45  DEF_60_CNT_SOCIAL_CIRCLE     306490 non-null  float64
 46  DAYS_LAST_PHONE_CHANGE       307510 non-null  float64
 47  FLAG_DOCUMENT_2              307511 non-null  int64  
 48  FLAG_DOCUMENT_3              307511 non-null  int64  
 49  FLAG_DOCUMENT_4              307511 non-null  int64  
 50  FLAG_DOCUMENT_5              307511 non-null  int64  
 51  FLAG_DOCUMENT_6              307511 non-null  int64  
 52  FLAG_DOCUMENT_7              307511 non-null  int64  
 53  FLAG_DOCUMENT_8              307511 non-null  int64  
 54  FLAG_DOCUMENT_9              307511 non-null  int64  
 55  FLAG_DOCUMENT_10             307511 non-null  int64  
 56  FLAG_DOCUMENT_11             307511 non-null  int64  
 57  FLAG_DOCUMENT_12             307511 non-null  int64  
 58  FLAG_DOCUMENT_13             307511 non-null  int64  
 59  FLAG_DOCUMENT_14             307511 non-null  int64  
 60  FLAG_DOCUMENT_15             307511 non-null  int64  
 61  FLAG_DOCUMENT_16             307511 non-null  int64  
 62  FLAG_DOCUMENT_17             307511 non-null  int64  
 63  FLAG_DOCUMENT_18             307511 non-null  int64  
 64  FLAG_DOCUMENT_19             307511 non-null  int64  
 65  FLAG_DOCUMENT_20             307511 non-null  int64  
 66  FLAG_DOCUMENT_21             307511 non-null  int64  
 67  AMT_REQ_CREDIT_BUREAU_HOUR   265992 non-null  float64
 68  AMT_REQ_CREDIT_BUREAU_DAY    265992 non-null  float64
 69  AMT_REQ_CREDIT_BUREAU_WEEK   265992 non-null  float64
 70  AMT_REQ_CREDIT_BUREAU_MON    265992 non-null  float64
 71  AMT_REQ_CREDIT_BUREAU_QRT    265992 non-null  float64
 72  AMT_REQ_CREDIT_BUREAU_YEAR   265992 non-null  float64
dtypes: float64(21), int64(40), object(12)
memory usage: 171.3+ MB
""")

st.write('**In :blue[**Table 6**], there are still some variables that need to be handled with null value as follows:**')
st.info(
f"""
* **AMT_ANNUITY**: Loan annuity.
* **AMT_GOODS_PRICE**: For consumer loans it is the price of the goods for which the loan is given.
* **CNT_FAM_MEMBERS**: How many observation of client's social surroundings with observable 30/60 DPD (days past due) default.
* **OBS_X_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings with observable X = 30 or 60 DPD (days past due) default.
* **DEF_X_CNT_SOCIAL_CIRCLE**: How many observation of client's social surroundings defaulted on X = 30 or 60 DPD (days past due).
* **AMT_REQ_CREDIT_BUREAU_X**: Number of enquiries to Credit Bureau about the client one X = HOUR/DAY/WEEK/QRT/YEAR/MON before application.
* **DAYS_LAST_PHONE_CHANGE**: How many days before application did client change phone.
These features will be replaced by :red[0]  value. This way of changing will not change the meaning of features.
"""
)

#-------------
# "AMT_ANNUITY" is Loan annuity, so we replace "NULL" = 0 : the customer don't have loan annuity. 
cleanup_application_train["AMT_ANNUITY"].fillna(0,inplace = True)

# "AMT_GOODS_PRICE" is For consumer loans it is the price of the goods for which the loan is given, so we replace "NULL" = 0 : the customer don't have  consumer loans. 
cleanup_application_train["AMT_GOODS_PRICE"].fillna(0,inplace = True)

# "CNT_FAM_MEMBERS": How many family members does client have, replace "NULL" = 0
cleanup_application_train["CNT_FAM_MEMBERS"].fillna(0,inplace = True)

# "OBS_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE":How many observation of client's social surroundings with observable 30/60 DPD (days past due) default
cleanup_application_train["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)
cleanup_application_train["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)

# "DEF_30_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE": How many observation of client's social surroundings defaulted on 30 DPD (days past due) 
cleanup_application_train["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)
cleanup_application_train["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)

# AMT_REQ_CREDIT_BUREAU_X: Number of enquiries to Credit Bureau about the client one X before application
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0,inplace = True)
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(0,inplace = True)
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_WEEK"].fillna(0,inplace = True)
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0,inplace = True)
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0,inplace = True)  
cleanup_application_train["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0,inplace = True) 

# DAYS_LAST_PHONE_CHANGE: Not change the null value, null value is a customer do not change their phone number
cleanup_application_train["DAYS_LAST_PHONE_CHANGE"].fillna(0,inplace = True) 

# "NAME_TYPE_SUITE":Who was accompanying client when he was applying for the loan, so replace "NULL" by "Unaccompanied"
cleanup_application_train["NAME_TYPE_SUITE"].fillna("Unaccompanied",inplace = True)

# "OCCUPATION_TYPE": What kind of occupation does the client have, replace "NULL" by "NoInformation"
cleanup_application_train["OCCUPATION_TYPE"].fillna("NoInformation",inplace = True)
#-------------


# The chart for "NAME_TYPE_SUITE"
NAME_TYPE_SUITE_count = cleanup_application_train["NAME_TYPE_SUITE"].value_counts()
x = NAME_TYPE_SUITE_count.index
y = NAME_TYPE_SUITE_count.values


st.write('_**:blue[**Chart 1:**] The categories of "NAME_TYPE_SUITE"**_.')
arr= pd.DataFrame(y,x)
st.bar_chart(arr)

# The chart for "OCCUPATION_TYPE"
OCCUPATION_TYPE_count = cleanup_application_train["OCCUPATION_TYPE"].value_counts()
x = OCCUPATION_TYPE_count.index
y = OCCUPATION_TYPE_count.values
st.write('_**:blue[**Chart 2:**] The categories of "OCCUPATION_TYPE"**_.')
arr= pd.DataFrame(y,x)
st.bar_chart(arr)

st.info(
f"""
**Meanwhile**, 
* **NAME_TYPE_SUITE**: Who was accompanying client when he was applying for the loan, replaced  "NULL" by "Unaccompanied" category in the dataset (:blue[Chart 1]).
* **OCCUPATION_TYPE**: What kind of occupation does the client have, replaced "NULL" by the new category is "NoInformation".
"""
)

# Number of unique classes in each object column
unique_object_column = cleanup_application_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
st.write('_**:blue[**Table 7:**] Number of unique classes in each object column**._')
st.text(unique_object_column)

CODE_GENDER_count = cleanup_application_train["CODE_GENDER"].value_counts()
x = CODE_GENDER_count.index
y = CODE_GENDER_count.values


st.write('_**:blue[**Chart 3:**] The categories of "CODE_GENDER"**_.')
arr= pd.DataFrame(y,x)
st.bar_chart(arr)
# The removing  'CODE_GENDER'
st.warning('In :blue[**Table 7**]  and :blue[**Chart 3**], we can see **CODE_GENDER** in addition to the value **M/F**\
            also have the other value :red[**XNA**] with the number of rows is 4. Therefore, It is better to remove 4 rows.')

# Remove 4 rows with the value of CODE_GENDER is 'XNA'
cleanup_application_train = cleanup_application_train[cleanup_application_train['CODE_GENDER'] != 'XNA']

# The correlation index of variables
st.write('_**:blue[**Table 8:**] The pairwise correlation of all columns**_.')
st.write(cleanup_application_train.corr())


# Pandas profiling for the dataset
st.write('**c. Pandas profiling for the dataset.**')
st.write('The cleaning data for **"application_train.csv"** is completed, in order to have more analysis and information when building the model,\
         we will use **Pandas Profiling**.')
Report_train_data = cleanup_application_train.profile_report()
st_profile_report(Report_train_data)


# "application_test.csv" data
st.subheader('2. The data "application_test.csv".')
st.write('We continue to discover and clean up the test data (Out of time).')


# Read "application_test.csv" from a folder on the local computer
application_test = pd.read_csv(r'C:\Users\YTRAN\OneDrive - Università degli Studi di Verona\Programing and Database\Project\Data\application_test.csv')
st.write('_**:blue[**Table 9:**] The overview of "application_test.csv"**._')
st.write(application_test.head(10))
st.write('We perform the same cleaning steps as the train data, and removing the features were droped in the train data.')

application_test['DAYS_EMPLOYED'].replace(365243,np.nan,inplace = True)
# "AMT_ANNUITY" is Loan annuity, so we replace "NULL" = 0 : the customer don't have loan annuity. 
application_test["AMT_ANNUITY"].fillna(0,inplace = True)

# "AMT_GOODS_PRICE" is For consumer loans it is the price of the goods for which the loan is given, so we replace "NULL" = 0 : the customer don't have  consumer loans. 
application_test["AMT_GOODS_PRICE"].fillna(0,inplace = True)

# "CNT_FAM_MEMBERS": How many family members does client have, replace "NULL" = 0
application_test["CNT_FAM_MEMBERS"].fillna(0,inplace = True)

# "OBS_30_CNT_SOCIAL_CIRCLE, OBS_60_CNT_SOCIAL_CIRCLE":How many observation of client's social surroundings with observable 30/60 DPD (days past due) default
application_test["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)
application_test["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)

# "DEF_30_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE": How many observation of client's social surroundings defaulted on 30 DPD (days past due) 
application_test["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)
application_test["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0,inplace = True)

# AMT_REQ_CREDIT_BUREAU_X: Number of enquiries to Credit Bureau about the client one X before application
application_test["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0,inplace = True)
application_test["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(0,inplace = True)
application_test["AMT_REQ_CREDIT_BUREAU_WEEK"].fillna(0,inplace = True)
application_test["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0,inplace = True)
application_test["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0,inplace = True)  
application_test["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0,inplace = True) 

# DAYS_LAST_PHONE_CHANGE: Not change the null value, null value is a customer do not change their phone number
application_test["DAYS_LAST_PHONE_CHANGE"].fillna(0,inplace = True) 

# "NAME_TYPE_SUITE":Who was accompanying client when he was applying for the loan, so replace "NULL" by "Unaccompanied"
application_test["NAME_TYPE_SUITE"].fillna("Unaccompanied",inplace = True)

# "OCCUPATION_TYPE": What kind of occupation does the client have, replace "NULL" by "NoInformation"
application_test["OCCUPATION_TYPE"].fillna("NoInformation",inplace = True)

application_test = application_test.drop(columns = [i for i in per_nullvalue_greater40.index])

st.write('The test stage is a necessary stage in the process of model building.\
         We need another dataset to validate the model which builds from the train data (**"application_train.csv"**).\
         In this part, we will clean up  "application_test.csv" (Test data), and show the information of the test data. \
         We have the description of the test data after cleaning up such as:')


st.write('_:blue[**Table 10**]: **The information of the features in the test data.**_')
st.text("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48744 entries, 0 to 48743
Data columns (total 72 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   SK_ID_CURR                   48744 non-null  int64  
 1   NAME_CONTRACT_TYPE           48744 non-null  object 
 2   CODE_GENDER                  48744 non-null  object 
 3   FLAG_OWN_CAR                 48744 non-null  object 
 4   FLAG_OWN_REALTY              48744 non-null  object 
 5   CNT_CHILDREN                 48744 non-null  int64  
 6   AMT_INCOME_TOTAL             48744 non-null  float64
 7   AMT_CREDIT                   48744 non-null  float64
 8   AMT_ANNUITY                  48744 non-null  float64
 9   AMT_GOODS_PRICE              48744 non-null  float64
 10  NAME_TYPE_SUITE              48744 non-null  object 
 11  NAME_INCOME_TYPE             48744 non-null  object 
 12  NAME_EDUCATION_TYPE          48744 non-null  object 
 13  NAME_FAMILY_STATUS           48744 non-null  object 
 14  NAME_HOUSING_TYPE            48744 non-null  object 
 15  REGION_POPULATION_RELATIVE   48744 non-null  float64
 16  DAYS_BIRTH                   48744 non-null  int64  
 17  DAYS_EMPLOYED                39470 non-null  float64
 18  DAYS_REGISTRATION            48744 non-null  float64
 19  DAYS_ID_PUBLISH              48744 non-null  int64  
 20  FLAG_MOBIL                   48744 non-null  int64  
 21  FLAG_EMP_PHONE               48744 non-null  int64  
 22  FLAG_WORK_PHONE              48744 non-null  int64  
 23  FLAG_CONT_MOBILE             48744 non-null  int64  
 24  FLAG_PHONE                   48744 non-null  int64  
 25  FLAG_EMAIL                   48744 non-null  int64  
 26  OCCUPATION_TYPE              48744 non-null  object 
 27  CNT_FAM_MEMBERS              48744 non-null  float64
 28  REGION_RATING_CLIENT         48744 non-null  int64  
 29  REGION_RATING_CLIENT_W_CITY  48744 non-null  int64  
 30  WEEKDAY_APPR_PROCESS_START   48744 non-null  object 
 31  HOUR_APPR_PROCESS_START      48744 non-null  int64  
 32  REG_REGION_NOT_LIVE_REGION   48744 non-null  int64  
 33  REG_REGION_NOT_WORK_REGION   48744 non-null  int64  
 34  LIVE_REGION_NOT_WORK_REGION  48744 non-null  int64  
 35  REG_CITY_NOT_LIVE_CITY       48744 non-null  int64  
 36  REG_CITY_NOT_WORK_CITY       48744 non-null  int64  
 37  LIVE_CITY_NOT_WORK_CITY      48744 non-null  int64  
 38  ORGANIZATION_TYPE            48744 non-null  object 
 39  EXT_SOURCE_2                 48736 non-null  float64
 40  EXT_SOURCE_3                 40076 non-null  float64
 41  OBS_30_CNT_SOCIAL_CIRCLE     48744 non-null  float64
 42  DEF_30_CNT_SOCIAL_CIRCLE     48744 non-null  float64
 43  OBS_60_CNT_SOCIAL_CIRCLE     48744 non-null  float64
 44  DEF_60_CNT_SOCIAL_CIRCLE     48744 non-null  float64
 45  DAYS_LAST_PHONE_CHANGE       48744 non-null  float64
 46  FLAG_DOCUMENT_2              48744 non-null  int64  
 47  FLAG_DOCUMENT_3              48744 non-null  int64  
 48  FLAG_DOCUMENT_4              48744 non-null  int64  
 49  FLAG_DOCUMENT_5              48744 non-null  int64  
 50  FLAG_DOCUMENT_6              48744 non-null  int64  
 51  FLAG_DOCUMENT_7              48744 non-null  int64  
 52  FLAG_DOCUMENT_8              48744 non-null  int64  
 53  FLAG_DOCUMENT_9              48744 non-null  int64  
 54  FLAG_DOCUMENT_10             48744 non-null  int64  
 55  FLAG_DOCUMENT_11             48744 non-null  int64  
 56  FLAG_DOCUMENT_12             48744 non-null  int64  
 57  FLAG_DOCUMENT_13             48744 non-null  int64  
 58  FLAG_DOCUMENT_14             48744 non-null  int64  
 59  FLAG_DOCUMENT_15             48744 non-null  int64  
 60  FLAG_DOCUMENT_16             48744 non-null  int64  
 61  FLAG_DOCUMENT_17             48744 non-null  int64  
 62  FLAG_DOCUMENT_18             48744 non-null  int64  
 63  FLAG_DOCUMENT_19             48744 non-null  int64  
 64  FLAG_DOCUMENT_20             48744 non-null  int64  
 65  FLAG_DOCUMENT_21             48744 non-null  int64  
 66  AMT_REQ_CREDIT_BUREAU_HOUR   48744 non-null  float64
 67  AMT_REQ_CREDIT_BUREAU_DAY    48744 non-null  float64
 68  AMT_REQ_CREDIT_BUREAU_WEEK   48744 non-null  float64
 69  AMT_REQ_CREDIT_BUREAU_MON    48744 non-null  float64
 70  AMT_REQ_CREDIT_BUREAU_QRT    48744 non-null  float64
 71  AMT_REQ_CREDIT_BUREAU_YEAR   48744 non-null  float64
dtypes: float64(21), int64(39), object(12)
memory usage: 26.8+ MB
""")


st.write('_:blue[**Table 11**:] **The statistical information of the features in the test data.**_')
st.write(application_test.describe().T)

# The model that explains the data
st.subheader('II. The model that explains the data.')
st.write('**Scorecard model** is a model class that is applied in many fields such as \
         finance, business, and social management. In finance, the Scorecard model uses individual information \
         of customers to predict the ability to repay customers. Especially, based on credit score, financial institutions \
         can provide better products and services if the customer has a high credit score \
         and lower than the customer with a low credit score.')
st.write('The kind of the customer information that used to build **Scorecard model** including:')
st.markdown(
f"""
* **Demographic**: Information related to personal characteristics such as education level,\
      income, gender, age, occupation, marital status, family size, number of dependents, etc.
* **Credit history**: Customer loan history data is aggregated from all banks operating in the territory of a country into one data center. \
    The bank can cross-check customer's credit information from other banks.
* **Transaction**: The transaction history on a credit card or ATM card will \
    gauge any part of a customer's financial well-being.
* **Collateral Information**: The information that accompanies mortgage loans.
"""
)
st.write('The dataset of the project, the features focus on the kind of \
         information as **Demographic, Credit history, Collateral Information**.')
st.write('In :blue[**Table 8**], we can recognize variables capable of predicting the model such as **EXT_SOURCE_2, EXT_SOURCE_3** \
         , but the correlation coefficient is not high. In Scorecard model, WOE (weight of evidence) and IV (Information value) is one of the feature selection \
         and feature engineering techniques commonly applied in scorecard modeling. This method will rank the variables into strong, \
         medium, weak, no impact, etc. In addition, WOE is also the method to solve the null values/information. \
         Therefore, in the next part, we will research WOE and IV.')
st.write('In **Scorecard model**, **WOE** (weight of evidence) is one of the feature selection \
         and feature engineering techniques commonly applied in scorecard modeling. This method will rank the variables into strong, \
         medium, weak, no impact, etc. In addition, WOE is also the method to solve the null values/information. \
         Therefore, in the next part, we will research WOE.')
st.write('Moreover, **WOE** is also the method to solve the null values/information, so we will calcualte **WOE** indication.')
st.subheader('a. ObtimalBinning')
st.write('OptBinning is a library written in Python implementing a strict and mathematical programming formulation for solving the optimal binning problem for a binary,\
          continuous or multiclass target type, incorporating constraints not previously addressed.')
st.write('The binning method will group the values of feature become groups\
         and calcualte **WOE** indicator.**ObtimalBinning** will find the optimal groups that are able to \
         distinguish between good and bad customers. The code of **ObtimalBinning** is as follows:')
st.code(f"""
        # 1. Define the feature and target arrays
X = cleanup_application_train['CODE_GENDER']
Y = cleanup_application_train['TARGET']
# 2. Instantiate class and fit to train dataset
optb = OptimalBinning(name='CODE_GENDER', dtype="categorical")
optb.fit(X,Y)
# 3. To visualize the results table and plot
optb.binning_table.build()
optb.binning_table.plot(metric="woe")""")


# 1. Define the feature and target arrays
X = cleanup_application_train['CODE_GENDER']
Y = cleanup_application_train['TARGET']
# 2. Instantiate class and fit to train dataset
optb = OptimalBinning(name='CODE_GENDER', dtype="categorical")
optb.fit(X,Y)
# 3. To perform the binning of a dataset
X_binned = optb.transform(X)
# 4. To visualize the results table and plot
binning_table_CODE_GENDER = optb.binning_table.build()
binning_chart_CODE_GENDER = optb.binning_table.plot(metric="woe")

# Binning plot of CODE_GENDER
st.write('_**:blue[**Chart 4:**] The binning chart of "CODE_GENDER"**_.')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig = binning_chart_CODE_GENDER)

# Binning table of CODE_GENDER
st.write('_**:blue[**Table 12:**] The binning chart of "CODE_GENDER"**_.')
st.write(binning_table_CODE_GENDER)

st.info('In :blue[**Chart 4**] and :blue[**Table 12**] we can see the bins including **0 = Female** and **1 = Male** \
        of CODE_GENDER. In addition, **Non-event** is **target = 0** and **Event** is target = 1. We can see that, \
        WOE of Female is around :blue[**0.15**], Male is around :blue[**-0.25**]. These values reprent **Bin 0 (F)** \
        has a good feature in recognizing good profiles and  these values reprent **Bin 1 (M)** \
        has a good feature in recognizing bad profiles.')
st.write('To calcualte WOE and IV for all features, we will go to **Binning Process**.')


# Binning Process 
st.subheader('b. BinningProcess')
st.write('Binary variables (0,1) when executing **BinningProcess**, the algorithm will divide into segments from 0 to 1, \
         this is not true of the meaning of binary variables. Therefore, before executing **BinningProcess**, \
         we will substitute these binary values into characters.')
st.write('_**SK_ID_CURR** is ID of loan in our sample, so we remove it from the train data:_')

st.code(f"""
df = cleanup_application_train.copy()
df = df.drop(columns=['SK_ID_CURR']))""")
st.code(f"""
# The list of the binary variables
binary_variables = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL'\
 ,'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',\
'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','LIVE_CITY_NOT_WORK_CITY',\
'FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',\
'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',\
'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',\
'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
# Replace the values of the binary variables
for i in binary_variables:
    df[i] = df[i].replace((0,1),['N','Y'])
# Replace the values of REGION_RATING_CLIENT
df["REGION_RATING_CLIENT"] = df["REGION_RATING_CLIENT"].replace([1,2,3],['A','B','C'])
""")
# Copy file cleanup_application_train and drop SK_ID_CURR
df = cleanup_application_train.copy()
df = df.drop(columns=['SK_ID_CURR'])

# The list of the binary variables
binary_variables = ['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL'\
 ,'REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',\
'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','LIVE_CITY_NOT_WORK_CITY',\
'FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',\
'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',\
'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',\
'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']

# Replace the values of the binary variables
for i in binary_variables:
    df[i] = df[i].replace((0,1),['N','Y'])

# Replace the values of REGION_RATING_CLIENT
df["REGION_RATING_CLIENT"] = df["REGION_RATING_CLIENT"].replace([1,2,3],['A','B','C'])

st.write('The train data after replacing the binary values as follows:')

st.write('_**:blue[**Table 13:**] The overview of the train data after replacing.**_')
st.write(df.head(10))

st.write('The following code to build BinningProcess, set up Minimum of IV indicator is **0.01** to choose the features which have \
         high effectiveness for the model. After that, we can see the results of Binning and the chosen features\
         with other IVs.')

st.code('''
# Separate Y = Target, X = the varibales to build the model
target = ['TARGET']
variable_names = [x for x in list(df.columns) if x not in target]
X = df[variable_names].values
y = df['TARGET']
# Set up the Min and Max of IV indicators
selection_criteria = {"iv": {"min": 0.01, "max": 15}}
# Create Binning
binning_process = BinningProcess(variable_names, selection_criteria=selection_criteria)
binning_process.fit(X,y)
# Show binning results
for variable in variable_names:
    optb = binning_process.get_binned_variable(name = variable)
    optb.binning_table.build()
    optb.binning_table.plot()
  ''',  language='python')


# Separate Y = Target, X = the varibales to build the model
target = ['TARGET']
variable_names = [x for x in list(df.columns) if x not in target]
X = df[variable_names].values
y = df['TARGET']

# Set up the Min and Max of IV indicators
st.subheader('**Binning for features:**')
IV = st.selectbox(
     '**Min IV:**',
     (0.01, 0.02, 0.03, 0.04, 0.05, 0.06))
selection_criteria = {"iv": {"min": IV, "max": 15}}
# Create Binning
binning_process = BinningProcess(variable_names, selection_criteria=selection_criteria)
binning_process.fit(X,y)

# Show binning results
option = st.selectbox(
     '**Features/Columns:**',
     variable_names)
optb = binning_process.get_binned_variable(option)
st.write('**Binning table**:', optb.binning_table.build())
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = optb.binning_table.plot(metric="woe")
st.write('**Binning Chart:**')
st.pyplot(fig)
bin_info = binning_process.information()
bin_summary = binning_process.summary()
st.write('**Binning Summary:**')
st.table(bin_summary)

st.write('In the binning table, there are features/variables that have high IV such as:\
        **AMT_ANNUITY, AMT_CREDIT, DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION,\
          DAYS_LAST_PHONE_CHANGE, AMT_GOODS_PRICE, DAYS_ID_PUBLISH, EXT_SOURCE_2, EXT_SOURCE_3**.')

# The Logistic Regression
clf1 = Pipeline(steps=[('binning_process', binning_process),('standardscaler', StandardScaler()),
                      ('classifier',LogisticRegression(solver="lbfgs",max_iter = 1000))])

# Divide the train/test data with test train = 80%, test size = 20%.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,train_size = 0.8, random_state=42)
# fit model 
clf1.fit(X_train, y_train)
# Confusion matrix for model
y_pred1 = clf1.predict(X_test)
Result_model = classification_report(y_test, y_pred1)
#st.table(Result_model)

# Visualize ROC
#probs = clf1.predict_proba(X_test)
#preds = probs[:,1]
#fpr1, tpr1, threshold = roc_curve(y_test, preds)
#roc_auc1 = auc(fpr1, tpr1)
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc='lower right')
#plt.plot([0, 1], [0, 1],'k--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.plot(fpr1, tpr1, 'b', label='Binning+LR: AUC = {0:.2f}'.format(roc_auc1))
#plt.show()



# ROC and KS chart : pip install kds
import kds
y_prob1 = clf1.predict_proba(X_test)[:, 1]
# ROC chart
ROC =  kds.metrics.plot_cumulative_gain(y_test,y_prob1)
plt.show()
st.pyplot(ROC)

# KS Chart
KS = kds.metrics.plot_ks_statistic(y_test, y_prob1)
plt.show()
st.pyplot(KS)

# Classification_report
Result_model = classification_report(y_test, y_pred1)
st.write('**The information of the model with min IV = 0.04:**')
st.text(f"""
              precision    recall  f1-score   support

           0       0.92      0.97      0.95     56481
           1       0.16      0.06      0.09      5021

    accuracy                           0.90     61502
   macro avg       0.54      0.52      0.52     61502
weighted avg       0.86      0.90      0.88     61502
""" 
)

st.write('We completed the model. In the next step, we save the model and apply it for the test data (Out of time). The following result:')
st.code('''
#3. applying for application_test (real set)
import pickle
application_test = application_test.drop(columns=['SK_ID_CURR'])
application_test = application_test.values
# Save the model
pickle.dump(binning_process_1, open('binning.pkl', 'wb'))
pickle.dump(clf2, open('clf1_model.pkl', 'wb'))
# load the model
load_binning = pickle.load(open('binning.pkl', 'rb'))
load_model = pickle.load(open('clf1_model.pkl', 'rb'))
# Predict for the test data (Out of time)  
y_apptest_pred = load_model.predict(application_test)
y_apptest_pred = pd.DataFrame(y_apptest_pred)
  ''',  language='python')


st.code(f'''
The number of Good/Bad customers for the Out of time data:
0    47391
1     1353
dtype: int64
''')
st.code(f'''
The percentage of Good/Bad customers for the Out of time data:
0    0.972243
1    0.027757
dtype: float64
''')