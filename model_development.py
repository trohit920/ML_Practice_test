import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv('pitta_clean_df.csv')
df.head()

lm = LinearRegression()
lm
X = df[['engine-size']]
Y = df['normalized-losses']
lm.fit(X,Y)
print(lm.intercept_)
print(lm.coef_)
lm1 = LinearRegression()
X = df[['width']]
Y = df['normalized-losses']
lm1.fit(X,Y)
Yhat=lm1.predict(X)
Yhat[0:5]  
print(lm1.intercept_)
print(lm1.coef_)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['normalized-losses'])
print(lm.intercept_)
print(lm.coef_)

width=12
height=10
plt.figure(figsize = (width,height))
sns.regplot(x='horsepower',y='normalized-losses',data=df)
plt.ylim(0,)
plt.show()
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
df[['peak-rpm','highway-mpg','normalized-losses']].corr()
width = 12
height = 10
plt.figure(figsize=(width,height))
sns.residplot(x="peak-rpm", y="normalized-losses", data=df)
plt.ylim(0,)
plt.show()
Yhat = lm.predict(Z)
plt.figure(figsize=(width,height))
ax1=sns.distplot(df['normalized-losses'],hist=False,color="r",label="Actual Values")
sns.distplot(Yhat,hist=False,color="b",label="Fitted Values",ax=ax1)
plt.title("Actual vs Fitted Values for normalized-losses")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")
plt.show()
plt.close()

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for normalized-losses ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('normalized-losses ')

    plt.show()
    plt.close()

x = df['engine-size']
y = df['normalized-losses']
#Let's fit the polynomial using the function polyfit,then use the function to display the polynomial function.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPolly(p, x, y, 'engine-size')
np.polyfit(x, y, 3)
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Length')

pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape
Z_pr
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]
Input = [('scale',StandardScaler()),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z,y)
ypipe = pipe.predict(Z)
ypipe[0:10]
lm.fit(X, Y)
lm.score(X, Y)
Yhat=lm.predict(X)
Yhat[0:4]
mse = mean_squared_error(df['normalized-losses'], Yhat)
lm.fit(Z, df['normalized-losses'])
lm.score(Z, df['normalized-losses'])
Y_predict_multifit = lm.predict(Z)
mean_squared_error(df['normalized-losses'], Y_predict_multifit)
r_squared = r2_score(y, p(x))
mean_squared_error(df['normalized-losses'], p(x))
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat=lm.predict(new_input)
yhat[0:5]
plt.plot(new_input, yhat)
plt.show()