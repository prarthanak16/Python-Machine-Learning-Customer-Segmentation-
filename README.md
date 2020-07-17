<h1>Customer Segmentation using Python-Machine Learning</h1>

 <h2>What is Customer Segmentation ?</h2>
      <h4>Customer Segmentation is the practice of dividing a customer base into groups of individuals that are similar in specific ways relevant to marketing,such as age,gender,interests and spending habits. Companies employing customer segmentation operates under the fact that every customer is different and their marketing efforts would be better served if they target specific and small groups with messagesz that consumers would find it relevant and lead them to buy something. Companies also hope for deeper understanding of their customers' prferences and needs.</h4>
<h4>Customer Segmentation relies on identifying key differentiators that divide customers into groups that are targeted :</h4>
<h4>> Customers' demographics (age,gender,income,race,religion,family-size,ethnicity,education level) </h4>
<h4>> Customers' Geography (where they live and work)</h4>
<h4>> Customers' Psychography (social class,lifestyle and personality characteristics)</h4>
<h4>> Customers' Behavior (spending,consumption,usage and desired benefits)</h4>

<h2>Customer segmentation procedures</h2>
<h4>Customer segmentation, also called consumer segmentation or client segmentation, procedures include:</h4>

<h4>* Deciding what data will be collected and how it will be gathered</h4>
<h4>* Collecting data and integrating data from various sources</h4>
<h4>* Developing methods of data analysis for segmentation</h4>
<h4>* Establishing effective communication among relevant business units (such as marketing and customer service) about the segmentation</h4>
<h4>* Implementing applications to effectively deal with the data and respond to the information it provides</h4>

<h2>Benefits of Customer Segmentation</h2>
<h4>- Helps identify least and most profitable customers, thus helping the business to concentrate marketing activities on those most likely to buy your products or services</h4>
<h4>- Helps build loyal relationships with customers by developing and offering them the products and services they want</h4>
<h4>- Helps improve customer service and alsp maximize use of your resourses</h4>
<h4>- Helps improve or tweak products to meet customer requirements</h4>
<h4>- Helps increase profit by keeping costs down</h4>

![Test Image1](https://www.enterrasolutions.com/wp-content/uploads/2016/08/Segmentation-clear.png)

<h2>The Challenge</h2>
<h4>You are owning a Mall and from membership card you have got some basic information of the customers such as gender, age, annual income and spending score. With this information you can perform customer segmentation through which you can decide who are your target customers and give this sense to marketing team and plan the strategy accrdingly. </h4>

<h2>Data</h2>
<h4>Data set needed for this project will be available [here](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)</h4>

<h2>Code</h2>
<h4>First we are supposed to import all the libraries an which are needed for executing the following codes</h4>

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

```
<h4>Then read in the Mall_customers csv file as a DataFrame called cust.</h4>
 
 ```
 cust=pd.read_csv("Mall_Customers.csv")
 cust.head()
 
 ```
<h4>Then to know how many columns are there and type of each column we hav to do:</h4>

```
cust.info()

```
<h4>To compute the summary of stastics pertaining to DataFrame columns which gives summary of only numeric columns</h4>
 
```

cust.describe()

```
<h4>Constructing histogram based on age frequency.Here we get to know that number of customers who belong to particular age-groups</h4>

```

sns.set_style('whitegrid')
cust['Age'].hist(bins=30)
plt.xlabel('Age')

```

<h4>Construction of jointplot() allows you to match up two distplots for bivariate data.Here we consider Age and Annual Income.</h4>

 ```
 
 sns.jointplot(x='Age',y='Annual Income (k$)',data=cust)
 
 ```
 
 <h4>Another way of constructing jointplot() of kind "kde" by considering Age and Spending Score.</h4>
 
 ```
 
 sns.jointplot(x='Age',y='Spending Score (1-100)',data=cust,color='green',kind='kde')
 
 ```
 <h4>Then we have to do a boxplot for Annual Income and Spending score for better understanding of distribution range. Here we clealy come to know that distribution range of Spending score is more than Annual Income</h4>
 
 ```
 
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.boxplot(y=cust["Spending Score (1-100)"],color="brown")
plt.subplot(1,2,2)
sns.boxplot(y=cust["Annual Income (k$)"],color="yellow")
plt.show()

```
<h4>Next constructing an pairplot() which will plot pairwise relationships across an entire dataframe. Here we first drop the CustomerID because it is not needed then we plot for Age, Annual Income and Spending score based on Gender.</h4>

```

cust.drop(["CustomerID"], axis = 1, inplace=True)
sns.pairplot(cust,hue='Gender',palette='Set1')

```



