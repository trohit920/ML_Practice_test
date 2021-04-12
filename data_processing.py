import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

other_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
# other_path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(other_path, header=None)
print("The first 5 rows of the dataframe: ", df.head(5)) 

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns = headers
df.head(10)
df.dropna(subset=["normalized-losses"], axis=0)
print("COLUMN NAMES: ",df.columns)
print("++++++++++++++++Object type+++++++++++++++++++: \n", df.dtypes)
print(df.describe())
print(df.info)

df.replace("?", np.nan, inplace = True)
missing_data = df.isnull()
missing_data.head(5)
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("") 

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace=True)

avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of Bore Values:",avg_bore)
df["bore"].replace(np.nan,avg_bore,inplace=True)

avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of Stroke Values:",avg_bore)
df["stroke"].replace(np.nan,avg_stroke,inplace = True)

avg_horsepower = df['horsepower'].astype("float").mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
df['num-of-doors'].replace(np.nan,"four",inplace=True)

df.dropna(subset=["normalized-losses"],axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")

df["city-L/100km"] = 235 / df["city-mpg"]
df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()
df['height'] = df['height']/df['height'].max() 
df[["length","width","height"]].head()

dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis = 1, inplace=True)
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'std':"aspiration-std",'turbo':"aspiration-turbo"},inplace=True)
dummy_variable_2.head()
df = pd.concat([df,dummy_variable_2],axis=1)
df.drop('aspiration',axis=1,inplace=True)
df.to_csv('pitta_clean_df.csv')

df = pd.read_csv('pitta_clean_df.csv')
print("+++++++++++++++++=Correlation++++++++++++++++++++++++++: \n ", df.corr())
df[['bore','stroke','compression-ratio','horsepower']].corr()
sns.regplot(x="engine-size", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
df[["engine-size", "normalized-losses"]].corr()
sns.regplot(x="highway-mpg", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
df[['highway-mpg', 'normalized-losses']].corr()
sns.regplot(x="peak-rpm", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
df[['peak-rpm','normalized-losses']].corr()
df[["stroke","normalized-losses"]].corr()
sns.regplot(x="stroke", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
sns.boxplot(x="body-style", y="normalized-losses", data=df)
plt.show()
sns.boxplot(x="engine-location", y="normalized-losses", data=df)
plt.show()
sns.boxplot(x="drive-wheels", y="normalized-losses", data=df)
plt.show()
print("++++++++++++++Description+++++++++++++++++: \n", df.describe())
print("++++++++++++++++++Description with object type+++++++++++++++++++++++: \n", df.describe(include=['object']))
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'},inplace=True)
drive_wheels_counts.index.name = "drive_wheels"
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
df_group_one = df[['drive-wheels','body-style','normalized-losses']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_gptest = df[['drive-wheels','body-style','normalized-losses']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
df_gptest2 = df[['body-style','normalized-losses']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index=False).mean()
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
plt.xticks(rotation=90)
fig.colorbar(im)
plt.show()

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between wheel-base and normalized-losses is", pearson_coef, " with a P-value of P =", p_value)  
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between horsepower and normalized-losses is", pearson_coef, " with a P-value of P = ", p_value)  
pearson_coef, p_value = stats.pearsonr(df['length'], df['normalized-losses'])
print("The Pearson Correlation Coefficient length and normalized-losses is", pearson_coef, " with a P-value of P = ", p_value)  
pearson_coef, p_value = stats.pearsonr(df['width'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between with and normalized-losses is", pearson_coef, " with a P-value of P =", p_value ) 
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['normalized-losses'])
print( "The Pearson Correlation Coefficient between curb-weight and normalized-losses is", pearson_coef, " with a P-value of P = ", p_value)  
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between engine-size and normalized-losses is", pearson_coef, " with a P-value of P =", p_value) 
pearson_coef, p_value = stats.pearsonr(df['bore'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between bore and normalized-losses  is", pearson_coef, " with a P-value of P =  ", p_value )
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['normalized-losses'])
print("The Pearson Correlation Coefficient between city-mpg and normalized-losses is", pearson_coef, " with a P-value of P = ", p_value) 
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['normalized-losses'])
print( "The Pearson Correlation Coefficient between highway-mpg and normalized-losses is", pearson_coef, " with a P-value of P = ", p_value ) 
grouped_test2=df_gptest[['drive-wheels', 'normalized-losses']].groupby(['drive-wheels'])
grouped_test2.head(2)
df_gptest
grouped_test2.get_group('4wd')['normalized-losses']
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['normalized-losses'], grouped_test2.get_group('rwd')['normalized-losses'], grouped_test2.get_group('4wd')['normalized-losses'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val) 
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['normalized-losses'], grouped_test2.get_group('rwd')['normalized-losses'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['normalized-losses'], grouped_test2.get_group('rwd')['normalized-losses'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['normalized-losses'], grouped_test2.get_group('fwd')['normalized-losses'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val) 

