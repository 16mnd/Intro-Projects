
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor



df = pd.read_csv("G:\\Multicollinearity\\House Sales.csv")



df.head()


type(df)


# ### Calculating VIF scores for original data




# Creating a function to calculate the VIF scores for all independant features with for loop


def vif_scores(df):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independent Features"] = df.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return VIF_Scores

df1 = df.iloc[:,:-1]
vif_scores(df1)


# ### Fixing Multicollinearity - dropping variables




#Copying the original dataframe
df2 = df.copy()





# Dropping the features which are having high VIF values
df3 = df2.drop(['Interior(Sq Ft)','# of Rooms'], axis = 1)





df3.head()





#Calculating VIF scores after dropping the varaibles
def vif_scores(df3):
    VIF_Scores = pd.DataFrame()
    VIF_Scores["Independant Features"] = df3.columns
    VIF_Scores["VIF Scores"] = [variance_inflation_factor(df3.values,i) for i in range(df3.shape[1])]
    return VIF_Scores

df3 = df3.iloc[:,:-1]
vif_scores(df3)


# ### Fixing multicollinearity - Combining the variables


df4= df3.copy()


df4.head()



#Combining the variables and calculating the VIF scores
df5 = df4.copy()
df5['Total Rooms'] = df4.apply(lambda x: x['# of Bed'] + x['# of Bath'],axis=1)
X = df5.drop(['# of Bed','# of Bath'],axis=1)
vif_scores(X)
