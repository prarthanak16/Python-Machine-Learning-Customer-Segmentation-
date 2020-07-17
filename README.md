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
<h4>Next we do a Bar plot to check the distribution of Male and Female in the dataset. Its shows that female population is more than the male population which gives a data on whom to target more to create an oppotunity to increase the marketing sales.</h4>

```
gender=cust.Gender.value_counts()
sns.barplot(x=gender.index,y=gender.values)

```

<h4>Next plot a piechart to know the population of male and female in terms of percentage. </h4>

```

fig = plt.figure()
pct=round(gender/sum(gender)*100)
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')

lbs=['Female','Male']
ax.pie(pct,labels=lbs,autopct='%1.2f%%')
plt.show()

```
<h4>Plotting the Barplot to know the distribution of Customers in each age-group. From this we come to know that the customers are more of age 31-40 so they should be more focused for marketing.</h4>

```

age0=cust.Age[(cust.Age<20)]
age1= cust.Age[(cust.Age>=20)&(cust.Age<=30)]
age2=cust.Age[(cust.Age>30)&(cust.Age<=40)]
age3=cust.Age[(cust.Age>40)&(cust.Age<=50)]
age4=cust.Age[(cust.Age>50)&(cust.Age<=60)]
age5=cust.Age[(cust.Age>60)]
x=['Below 20','21-30','31-40','41-50','51-60','60+']
y=[len(age0.values),len(age1.values),len(age2.values),len(age3.values),len(age4.values),len(age5.values)]
sns.barplot(x=x,y=y,palette='Accent')
plt.title(str("Number of customers based on Age Group"))
plt.xlabel(str("Age"))
plt.ylabel(str("Number of customers"))
plt.show()

```
<h4>Then we make a bar plot to visualize the number of customers according to their annual income. The majority of the customers have annual income in the range Rs60000 and Rs75000.</h4>

```

plt.figure(figsize=(15,7))
income0=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]<15)]
income1=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>=15)&(cust["Annual Income (k$)"]<=30)]
income2=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>30)&(cust["Annual Income (k$)"]<=45)]
income3=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>45)&(cust["Annual Income (k$)"]<=60)]
income4=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>60)&(cust["Annual Income (k$)"]<=75)]
income5=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>75)&(cust["Annual Income (k$)"]<=90)]
income6=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>90)&(cust["Annual Income (k$)"]<=105)]
income7=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>105)&(cust["Annual Income (k$)"]<=120)]
income8=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>120)&(cust["Annual Income (k$)"]<=135)]
income9=cust["Annual Income (k$)"][(cust["Annual Income (k$)"]>135)&(cust["Annual Income (k$)"]<=150)]
x=['Below 15','15-30','31-45','46-60','61-75','76-90','91-105','106-120','121-135','136-150']
y=[len(income0.values),len(income1.values),len(income2.values),len(income3.values),len(income4.values),len(income5.values),len(income6.values),len(income7.values),len(income8.values),len(income9.values)]
sns.barplot(x=x,y=y)

plt.title("Number of Customers based on Annual Income")
plt.xlabel("Annual income(k$)")
plt.ylabel("Number of customers")

```
<h4>Next continue with making a bar plot to visualize the number of customers according to their spending scores. The majority of the customers have spending score in the range 41–60.</h4>

```

spend0=cust["Spending Score (1-100)"][(cust["Spending Score (1-100)"]>0)&(cust["Spending Score (1-100)"]<=20)]
spend1=cust["Spending Score (1-100)"][(cust["Spending Score (1-100)"]>20)&(cust["Spending Score (1-100)"]<=40)]
spend2=cust["Spending Score (1-100)"][(cust["Spending Score (1-100)"]>40)&(cust["Spending Score (1-100)"]<=60)]
spend3=cust["Spending Score (1-100)"][(cust["Spending Score (1-100)"]>60)&(cust["Spending Score (1-100)"]<=80)]
spend4=cust["Spending Score (1-100)"][(cust["Spending Score (1-100)"]>80)&(cust["Spending Score (1-100)"]<=100)]
x=['0-20','21-40','41-60','61-80','81-100']
y=[len(spend0.values),len(spend1.values),len(spend2.values),len(spend3.values),len(spend4.values)]
sns.barplot(x=x,y=y,palette='gist_rainbow')

```
<h3>Decision Tree Classification<h3>
 
 <h4>Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.</h4>
 <h4>Here we use Decision tree Classifier for getting precision based on Gender</h4>
 
 <h4>Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.30</h4>
 
 ```
 
 from sklearn.model_selection import train_test_split
 
 ```
 <h4>X is the data with independent variables and y is the data with dependent variables</h4>
 
 ```
 X=cust.drop('Gender',axis=1)
 y=cust['Gender']
 
 ```
 
 ```
 
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
 
 ```
 <h4>Training the model:</h4>
 <h4>Importing DecisionTreeClassifier from sklearn.Tree</h4>
 
 ```
 
 from sklearn.tree import DecisionTreeClassifier
 ```
 <h4>Creating an Instance of DecisionTreeClassifier() named as dtree
 
 ```
 
 dtree=DecisionTreeClassifier()
 
 ```
 <h4>We’re going to use x_train and y_train, obtained above, to train our decision tree classifier. We’re using the fit method and passing the parameters as shown below.</h4>
 
 ```
 
 dtree.fit(X_train,y_train)
 
 ```
 <h4>Once the model is trained, it’s ready to make predictions. We can use the predict method on the model and pass x_test as a parameter to get the output as y_pred.</h4>
 
 ```
 
 predictions=dtree.predict(X_test)
 
 ```
 <h4>The classification report shows a representation of the main classification metrics on a per-class basis and the Confusion matrix is a summary of prediction results on a classification problem.</h4>
 <h4>We have to import classification_report and confusion_matrix from sklearn.metrics</h4>
 
 ```from sklearn.metrics import classification_report,confusion_matrix
 
 ```
 ```
 print(classification_report(y_test,predictions))
 
 ```
```

print(confusion_matrix(y_test,predictions))

```
<h2>K-means Clustering </h2>
<h4>K-means clustering is a clustering algorithm that aims to partition n observations into k clusters.</h4>

<h3>There are 3 steps:</h3>
<h4>* Initialisation – K initial “means” (centroids) are generated at random</h4>
<h4> * Assignment – K clusters are created by associating each observation with the nearest centroid</h4>
<h4>* Update – The centroid of the clusters becomes the new mean</h4>
 

<h4>Assignment and Update are repeated iteratively until convergence</h4>

<h4>The end result is that the sum of squared errors is minimised between points and their respective centroids.</h4>

<h2>Within Cluster Sum Of Squares (WCSS)</h2>
<h4>WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids.</h4>

![Test1 Image2](https://miro.medium.com/max/301/0*_3RAyFi3C2zJ-ShA.png)

 
 
 






