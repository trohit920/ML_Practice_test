#!/usr/bin/env python
# coding: utf-8

# <h1>Model Development</h1>

# <p>Lets develop several models that will predict the price of the car using the variables or features. This is just an estimate but should give us an objective idea of how much the car should cost.</p>

# 
# <p>We often use <b>Model Development</b> to help us predict future observations from the data we have.</p>
# 
# <p>A Model will help us understand the exact relationship between different variables and how these variables are used to predict the result.</p>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[108]:


# path of data 


# <h3> Linear Regression and Multiple Linear Regression</h3>

# <h4>Linear Regression</h4>

# 
# <p>One example of a Data  Model that we will be using is</p>
# <b>Simple Linear Regression</b>.
# 
# <br>
# <p>Simple Linear Regression is a method to help us understand the relationship between two variables:</p>
# <ul>
#     <li>The predictor/independent variable (X)</li>
#     <li>The response/dependent variable (that we want to predict)(Y)</li>
# </ul>
# 
# <p>The result of Linear Regression is a <b>linear function</b> that predicts the response (dependent) variable as a function of the predictor (independent) variable.</p>
# 
# 

# $$
#  Y: Response \ Variable\\
#  X: Predictor \ Variables
# $$
# 

#  <b>Linear function:</b>
# $$
# Yhat = a + b  X
# $$

# <ul>
#     <li>a refers to the <b>intercept</b> of the regression line, in other words: the value of Y when X is 0</li>
#     <li>b refers to the <b>slope</b> of the regression line, in other words: the value with which Y changes when X increases by 1 unit</li>
# </ul>

# In[5]:


#Lets load the modules for linear regression
from sklearn.linear_model import LinearRegression
#Create the linear regression object
lm = LinearRegression()
lm


# Lets create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable to see whether highway-mpg would help us to predict the car price

# In[6]:


X = df[['highway-mpg']]
Y = df['price']
#Fit the linear model using highway-mpg
lm.fit(X,Y)


#  We can output a prediction 

# In[7]:


Yhat=lm.predict(X)
Yhat[0:5]   


# <h4>Lets get the value of intercept (a) and Slope (b)</h4>

# In[12]:


print(lm.intercept_)
print(lm.coef_)


# Plugging in the actual values we get:

# <b>price</b> = 38423.31 - 821.73 x  <b>highway-mpg</b>

# Lets create a linear function with "engine-size" as the predictor variable and the "price" as the response variable to see whether engine-size would help us to predict the car price

# In[13]:


lm1 = LinearRegression()
X = df[['engine-size']]
Y = df['price']
lm1.fit(X,Y)


# In[117]:


Yhat=lm1.predict(X)
Yhat[0:5]   


# In[14]:


print(lm1.intercept_)
print(lm1.coef_)


# Plugging in the values ,we get
# 
# <b>price</b> = -7963.338906281049 + 166.86001569 x <b>engine-size</b>

# <h4>Multiple Linear Regression</h4>

# <p>This method is used to explain the relationship between one continuous response (dependent) variable and <b>two or more</b> predictor (independent) variables.
# Most of the real-world regression models involve multiple predictors.

# $$
# Y: Response \ Variable\\
# X_1 :Predictor\ Variable \ 1\\
# X_2: Predictor\ Variable \ 2\\
# X_3: Predictor\ Variable \ 3\\
# X_4: Predictor\ Variable \ 4\\
# $$

# $$
# a: intercept\\
# b_1 :coefficients \ of\ Variable \ 1\\
# b_2: coefficients \ of\ Variable \ 2\\
# b_3: coefficients \ of\ Variable \ 3\\
# b_4: coefficients \ of\ Variable \ 4\\
# $$

# The equation is given by

# $$
# Yhat = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$

# <p>From the previous section  we know that other good predictors of price could be:</p>
# <ul>
#     <li>Horsepower</li>
#     <li>Curb-weight</li>
#     <li>Engine-size</li>
#     <li>Highway-mpg</li>
# </ul>
# Let's develop a model using these variables as the predictor variables.

# In[15]:


Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
#Fit the linear model
lm.fit(Z, df['price'])


# In[16]:


print(lm.intercept_)
print(lm.coef_)


# <b>Price</b> = -15678.742628061467 + 52.65851272 x <b>horsepower</b> + 4.69878948 x <b>curb-weight</b> + 81.95906216 x <b>engine-size</b> + 33.58258185 x <b>highway-mpg</b>

# <h3>  Model Evaluation using Visualization</h3>

# To evaluate our models and to choose the best one? One way to do this is by using visualization.

# In[18]:


# import the visualization package: seaborn
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <h3>Regression Plot</h3>

# <p>When it comes to simple linear regression, an excellent way to visualize the fit of our model is by using <b>regression plots</b>.</p>
# 
# <p>This plot will show a combination of a scattered data points (a <b>scatter plot</b>), as well as the fitted <b>linear regression</b> line going through the data. This will give us a reasonable estimate of the relationship between the two variables, the strength of the correlation, as well as the direction (positive or negative correlation).</p>

#  Let's visualize Horsepower as potential predictor variable of price:

# In[19]:


width=12
height=10
plt.figure(figsize = (width,height))
sns.regplot(x='horsepower',y='price',data=df)
plt.ylim(0,)


# <p>We can see from this plot that <b>price is negatively correlated to highway-mpg, since the regression slope is negative</b>
# One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data, and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data. 
#     
# <p>Let's compare this plot to the regression plot of "peak-rpm".</p>

# In[179]:


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# <p>Comparing the regression plot of "peak-rpm" and "highway-mpg" we see that the points for "highway-mpg" are much closer to the generated line and on the average decrease. The points for "peak-rpm" have more spread around the predicted line, and it is much harder to determine if the points are decreasing or increasing as the "highway-mpg" increases.</p>

# In[180]:


df[['peak-rpm','highway-mpg','price']].corr()


# The variable "peak-rpm" has a stronger correlation with "price", it is approximate -0.704692  compared to   "highway-mpg" which is approximate -0.101616.

# <h3>Residual Plot</h3>
# 
# <p>A good way to visualize the variance of the data is to use a residual plot.</p>
# 
# <p><b>Residual</b> : The difference between the observed value (y) and the predicted value. It is the distance from the data point to the fitted regression line. (Yhat) is called the residual</p>
# 
# <p><b>Residual plot</b> : It is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.</p>
# <p>We should always look at the spread of the residuals:</p>
# 
# <p>- If the points in a residual plot are <b>randomly spread out around the x-axis</b>, then a <b>linear model is appropriate</b> for the data ( Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data )

# In[181]:


width = 12
height = 10
plt.figure(figsize=(width,height))
sns.residplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# <p>We can see from this residual plot - residuals are not randomly spread around the x-axis,thus a non-linear model is more appropriate for this data.</p>

# <h3>Multiple Linear Regression</h3>

# <p>Visualizing a model for Multiple Linear Regression<p>
# <p><b> Distribution plot</b> : Compare the distribution of the fitted values that result from the model and distribution of the actual values.</p>

# In[21]:


Yhat = lm.predict(Z)
plt.figure(figsize=(width,height))
ax1=sns.distplot(df['price'],hist=False,color="r",label="Actual Values")
sns.distplot(Yhat,hist=False,color="b",label="Fitted Values",ax=ax1)

plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")

plt.show()
plt.close()


# <p>We can see that the fitted values are reasonably close to the actual values, since the two distributions overlap a bit. However, there is definitely some room for improvement.</p>

# <h2>Polynomial Regression and Pipelines</h2>

# <p><b>Polynomial regression</b> is a particular case of the general linear regression model or multiple linear regression models.
# <p>We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.
# <p>There are different orders of polynomial regression:</p>

# <center><b>Quadratic - 2nd order</b></center>
# $$
# Yhat = a + b_1 X^2 +b_2 X^2 
# $$
# 
# 
# <center><b>Cubic - 3rd order</b></center>
# $$
# Yhat = a + b_1 X^2 +b_2 X^2 +b_3 X^3\\
# $$
# 
# 
# <center><b>Higher order</b>:</center>
# $$
# Y = a + b_1 X^2 +b_2 X^2 +b_3 X^3 ....\\
# $$

# <p>A linear model did not provide the best fit while using highway-mpg as the predictor variable. Let's see if we can try fitting a polynomial model to the data instead.</p>

# In[27]:


#We will use the following function to plot the data:
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[28]:


x = df['highway-mpg']
y = df['price']
#Let's fit the polynomial using the function polyfit,then use the function to display the polynomial function.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


#  Let's plot the function 

# In[29]:


PlotPolly(p, x, y, 'highway-mpg')


# In[192]:


np.polyfit(x, y, 3)


# <p>We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function  "hits" more of the data points.</p>

# In[30]:


#Create 11 order polynomial model with the variables x and y from above
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Length')


# <p>The analytical expression for Multivariate Polynomial function gets complicated. For example, the expression for a second-order (degree=2)polynomial with two variables is given by:</p>

# $$
# Yhat = a + b_1 X_1 +b_2 X_2 +b_3 X_1 X_2+b_4 X_1^2+b_5 X_2^2
# $$

# We can perform a polynomial transform on multiple features. 

# In[31]:


#import the module
from sklearn.preprocessing import PolynomialFeatures
#create a PolynomialFeatures object of degree 2 
pr=PolynomialFeatures(degree=2)
pr


# In[32]:


Z_pr=pr.fit_transform(Z)


# The original data is of 201 samples and 4 features 

# In[33]:


Z.shape


# After the transformation, there are 201 samples and 15 features

# In[34]:


Z_pr.shape


# <h2>Pipeline</h2>

# <p>Data Pipelines simplify the steps of processing the data.

# In[35]:


#Use the module Pipeline to create a pipeline and also use StandardScaler as a step in our pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[36]:


#create the pipeline by creating a list of tuples including the name of the model/estimator & its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[37]:


#Input the list as an argument to the pipeline constructor 
pipe=Pipeline(Input)
pipe


# In[38]:


#Normalize the data and perform a transform and fit the model simultaneously
pipe.fit(Z,y)


# In[39]:


#Normalize the data, perform a transform and produce a prediction  simultaneously
ypipe=pipe.predict(Z)
ypipe[0:4]


# <p> Lets create a pipeline that Standardizes the data, then perform prediction using a linear regression model using the features Z and targets y</p>

# In[40]:


Input = [('scale',StandardScaler()),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(Z,y)
ypipe = pipe.predict(Z)
ypipe[0:10]


# <h2> Measures for In-Sample Evaluation</h2>

# <p>When evaluating our models, not only do we want to visualize the results, but we also want a quantitative measure to determine how accurate the model is.</p>
# 
# <p>Two very important measures that are often used in Statistics to determine the accuracy of a model are:</p>
# <ul>
#     <li><b>R^2 / R-squared</b></li>
#     <li><b>Mean Squared Error (MSE)</b></li>
# </ul>
#     
# <b>R-squared</b> : R squared, also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line.The value of the R-squared is the percentage of variation of the response variable (y) that is explained by a linear model.</p>
# 
# <b>Mean Squared Error (MSE)</b> : The Mean Squared Error measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).</p>

# <h3>Model 1: Simple Linear Regression</h3>

# * Let's calculate the R^2

# In[44]:


#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
lm.score(X, Y)


# We can say that ~ 49.659% of the variation of the price is explained by this simple linear model "horsepower_fit".

# * Let's calculate the MSE

# In[45]:


#We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
Yhat=lm.predict(X)
Yhat[0:4]


# In[47]:


#import the function mean_squared_error from the module metrics
from sklearn.metrics import mean_squared_error
#compare the predicted results with the actual results
mse = mean_squared_error(df['price'], Yhat)
mse


# <h3>Model 2: Multiple Linear Regression</h3>

# * Let's calculate the R^2

# In[49]:


# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
lm.score(Z, df['price'])


# We can say that ~ 80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".

# * Let's calculate the MSE

# In[50]:


# Produce a prediction 
Y_predict_multifit = lm.predict(Z)
# Compare the predicted results with the actual results
# The mean square error of price and predicted value using multifit is: 
mean_squared_error(df['price'], Y_predict_multifit)


# <h3>Model 3: Polynomial Fit</h3>

# * Let's calculate the R^2

# let’s import the function <b>r2_score</b> from the module <b>metrics</b> as we are using a different function

# In[51]:


from sklearn.metrics import r2_score


# We apply the function to get the value of r^2

# In[52]:


r_squared = r2_score(y, p(x))
r_squared


# We can say that ~ 67.419 % of the variation of price is explained by this polynomial fit

# <h3>MSE</h3>

# We can also calculate the MSE:  

# In[53]:


mean_squared_error(df['price'], p(x))


# <h2>Prediction and Decision Making</h2>
# <h3>Prediction</h3>
# 
# <p>We trained the model using the method <b>fit</b>. Now we will use the method <b>predict</b> to produce a prediction.</p>
# <p>Lets import <b>pyplot</b> for plotting; we will also be using some functions from numpy.</p>

# In[54]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# Create a new input 

# In[55]:


new_input=np.arange(1, 100, 1).reshape(-1, 1)


#  Fit the model 

# In[56]:


lm.fit(X, Y)
lm


# Produce a prediction

# In[57]:


yhat=lm.predict(new_input)
yhat[0:5]


# Plot the data 

# In[58]:


plt.plot(new_input, yhat)
plt.show()


# <h3>Decision Making: Determining a Good Model Fit</h3>

# * <b>Model with the higher R-squared value is a better fit</b> for the data.
# 
# * <b>Model with the smallest MSE value is a better fit</b> for the data.</p>
# 
# 
# <h4>Let's take a look at the values for the different models.</h4>
# 
# <p><b>Simple Linear Regression</b> : Using Highway-mpg as a Predictor Variable of Price.
# <ul>
#     <li>R-squared: 0.49659118843391759</li>
#     <li>MSE: 3.16 x10^7</li>
# </ul>
# </p>
#     
# <p><b>Multiple Linear Regression</b> : Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.
# <ul>
#     <li>R-squared: 0.80896354913783497</li>
#     <li>MSE: 1.2 x10^7</li>
# </ul>
# </p>
#     
# <p><b>Polynomial Fit</b> : Using Highway-mpg as a Predictor Variable of Price.
# <ul>
#     <li>R-squared: 0.6741946663906514</li>
#     <li>MSE: 2.05 x 10^7</li>
# </ul>
# </p>

# <h2>Conclusion:</h2>

# <p>Comparing these three models, we conclude that <b>the MLR model is the best model</b> to be able to predict price from our dataset. This result makes sense, since we have 27 variables in total, and we know that more than one of those variables are potential predictors of the final car price.</p>
