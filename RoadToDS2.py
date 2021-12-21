#So, now I will skip to more advanced tools for Data Science. The databases I will be working with, are samples of
##Used Car sales in the UK or something as such by brand. Yeh, sales of used cars somewhere in the UK, or of a reseller
###or whatever. Sales. We will first check the BMW used car sales.


#I will do an analysis about how every brand is valuated over time, given it's model, year, transmission, mileage, fuel-
##type mpg and engine size

#I use Pandas to read the document, because Panda is life.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.linear_model import LinearRegression
import researchpy as rp

from sklearn.preprocessing import scale
sales_bmw='C:/Users/CrsTn/Documents/GitHub/RoadToDS2/Used Car Sales Database/bmw.csv'



bmw=pd.read_csv(sales_bmw)
type(bmw)
bmw.info()

#Everything is now set, we will check the column names to see what is the kind of data we are going to analyze and drop NA's
bmw.columns
bmw.dropna()

#One of the first steps I will take to make things more efficient, is to change some data type. Some numbers are small
##enough to not be stored as float64 or int64, so we want to transform then into float16 or int16 for example to make the dataset
###smaller and more efficient to analyse for the PC.
bmw.dtypes


bmw.max() #As we can see here, engine size could support float16 as the biggest value is smaller than 127. Tax and year can be converted
            ##into int16, just like mpg  can be converted into float16

bmw.min() #From this command we can see the i Series do not have engine size measured as usual, as they're electric cars.

#To do so we will use the astype provided by panda
bmw.info() #Data usage is now over 758 KB

bmw['engineSize']=bmw['engineSize'].astype(np.float16)
bmw['mpg']=bmw['mpg'].astype(np.float16)

bmw['tax']=bmw['tax'].astype(np.int16)
bmw['year']=bmw['year'].astype(np.int16)

bmw['mileage']=bmw['mileage'].astype(np.int32)

bmw.info() ##If we check the used memory, we will see it's about 460 KBs now, which is substantially less.
                                        #####Linear Models#####
#One assumption of linear regressions is  Independence of the independent variables, thus we will check the correlation
##between the different columns in the dataset now. For this, we will create a correlationMatrix that will be converted
### into heatmaps for pretty visualizations

from collections import Counter
type(bmw[])
c= Counter(bmw['model']) ##This will count all sales by car model.

print(c)

corrbmw=bmw.corr()
sb.heatmap(corrbmw, annot=True) #The most correlation that exists here is above 0.7, but bellow 0.8. Mileage will clearly
                                    ##Be correlated with the price, but it is still a relevant variable we will need. Just like
                                    ###The year of the car.
                                    ####Tax and MPG seem not to have any linear relation with, so we will drop them later.


#Another assumption is a Linear Correlation between the predicate and the predictant. Seaborn provides us with useful
##Tools to check this.


#We will now prepare the dataset for linear regression. For this, we need to check for outliers, since the
##Non-existent values have been dropped from before. This is another of the Regression assumptions. There are no outliers


#As year, tax, mpg and engineSize is arbitrarily set, no outliers will be searched for
##here. Other is the story for price and mileage. Outliers for these will be dropped and we will do so with the
###scipy API
import scipy as sp

sb.boxplot(x=bmw['price'])
sb.histplot(bmw['price'])


sb.boxplot(x=bmw['mileage'])
sb.histplot(bmw['mileage'])

#Z-score will help us with this. The Z score tells you how many Standard deviations the observation is from the mean of
##the sample and 3 deviations from the mean are usually considered outliers.Anyways, I will use 3 AND 1.5 SD,
### to eliminate more extreme outliers by using 1.5 SD's forward or backward.


z_scores= pd.DataFrame(abs(sp.stats.zscore(bmw['price'])))
z_scores2=pd.DataFrame(abs(sp.stats.zscore(bmw['mileage'])))


z_scores3=pd.concat([z_scores, z_scores2], axis=1)

bmw.shape  #This command can be used to check later if the dataset has been made smaller
type(bmw)

#I wil use two different datasets from now on. One will be filtered with 3 times the z-score as maximum limit for
##outliers and then 1.5 times, to make the model a little less biased.



bmw_wo_outliers = bmw[(z_scores3<3).all(axis=1)]

bmw_wo_outliers2 = bmw[(z_scores3<1.5).all(axis=1)]

##We can check if this worked, by using the STD function to calculate Standard Deviation
np.std(bmw['price'])
np.std(bmw_wo_outliers['price'])
np.std(bmw_wo_outliers2['price'])



np.std(bmw['mileage'])
np.std(bmw_wo_outliers['mileage'])
np.std(bmw_wo_outliers2['mileage'])


sb.boxplot(x=bmw_wo_outliers['mileage'])


#Next, I'll do a T-test to check if there's a statistical difference between the means of the samples

from scipy.stats import ttest_ind

ttest_ind(bmw_wo_outliers['price'], bmw_wo_outliers['engineSize'], equal_var=False)

ttest_ind(bmw_wo_outliers['price'], bmw_wo_outliers['mpg'], equal_var=False)

ttest_ind(bmw_wo_outliers['price'], bmw_wo_outliers['tax'], equal_var=False)

ttest_ind(bmw_wo_outliers['price'], bmw_wo_outliers['mileage'], equal_var=False)

#All the p-values in these T-tests were all near to 0, so we will include all of the independent variables in our model
##to check the effect of them on Used cars value.

####plt.pyplot.table(cellText='c', cellColours=None, cellLoc='right', colWidths=None, rowLabels=None,
                       # rowColours=None, rowLoc='left', colLabels=None, colColours=None, colLoc='center', loc='bottom',
                        #bbox=None, edges='closed', **kwargs)


#We can check for outliers remaining in our dataset and some correlation visuals for... fun.

plt.plot(bmw['mileage'], bmw['price'], 'r^')
plt.xlabel('Miles driven')
plt.ylabel('Price')
plt.show

plt.clf()

plt.plot(bmw_wo_outliers['mileage'], bmw_wo_outliers['price'], 'r^')
plt.xlabel('Miles driven')
plt.ylabel('Price')
plt.show


plt.plot(bmw_wo_outliers2['mileage'], bmw_wo_outliers2['price'], 'r^')
plt.xlabel('Miles driven')
plt.ylabel('Price')
plt.show
#There's a visible correlation right here. The more miles driven, the lower the price (no sh#t, Sherlock). Therefore
##Our regression will be about price, because that was the info we wanted to know about. Price can give us a
###notion of how people value different cars over time and use, so it is useful to know.


                                    ####Simple Linear Regression####

x = bmw_wo_outliers['mileage']
x= x.array.reshape(-1,1)
y= bmw_wo_outliers['price']


x1= bmw_wo_outliers2['mileage']
x1= x1.array.reshape(-1,1)
y1= bmw_wo_outliers2['price']


LinReg = LinearRegression()

LinReg.fit(x,y)

LinReg2 = LinearRegression()
LinReg2.fit(x1,y1)
print("The intercept for our Z<3 is ", LinReg.intercept_,"and the Coefficient for Price is", LinReg.coef_, " \n While the intercept for Z<1.5 is", LinReg2.intercept_, "and the Price Coefficient is", LinReg2.coef_)

print(LinReg.score(x,y), "is the R squared for the first model and", LinReg2.score(x1,y1), "is the R squared for the second one")




                                       ####Multiple Linear Regression####

mlr_bmw=bmw_wo_outliers[['engineSize', 'mpg', 'tax', 'mileage']].values
mlr_bmw2=bmw_wo_outliers2[['engineSize', 'mpg', 'tax', 'mileage']].values





target_price=bmw_wo_outliers[['price']].values
target_price2=bmw_wo_outliers2[['price']].values

mlr_data_names=['engineSize', 'mpg', 'tax', 'mileage']
mlr_data_names2=['engineSize', 'mpg', 'tax', 'mileage']




X,y = scale(mlr_bmw), target_price
X2, y2= scale(mlr_bmw2), target_price2

LinReg= LinearRegression()
LinReg2= LinearRegression()


LinReg.fit(X,y)


LinReg.fit(X2,y2)

print(LinReg.score(X, y))

print(LinReg.score(X2, y2))


#I will leave the car sales database for a while, because I'm going to study Logistic Regressions in python,
##which needs binary or ordinal (categorical variable with ordered values) variables.

                                        ####Logistic Regressions####




    import numpy as np
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    import sklearn

    from pandas import Series, DataFrame
    from pylab import rcParams
    from sklearn import preprocessing


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

titan= 'C:/Users/CrsTn/Desktop/Ex_Files_Python_Data_Science_EssT_Pt2/Exercise Files/Data/titanic-training-data.csv'
titanic_training= pd.read_csv(titan)

titanic_training.columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket','Fare', 'Cabin', 'Embarked']

titanic_training.head()


print(titanic_training.info)

#Check for model assumptions, before I implement the model

sb.countplot(x='Survived', data=titanic_training, palette='hls') #Checking if the dependent variable is binary


titanic_training.isnull().sum() #Counting for missing values. We can see cabin has a lot of them and age does too
                                ## but age does matter more for this purpose.


titanic_training.describe()
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic_data.head

#To check if an approximation of the age is possible, I'm going to check for correlations with parch.
sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')

#To do the approximation, we need the average age

parch_groups= titanic_data.groupby(titanic_data['Parch']) #This creates means by the quantity of parent or children they
                                                          ## have aboard

parch_groups.mean()                         #This can be used to fill the age values that are missing.

titanic_data['Age'].describe()

from collections import Counter


def age_approx(cols):
    Age= cols[0]
    Parch= cols[1]

    if pd.isnull(Age):
        if Parch == 0:
            return 32 ##Average for 0 Parch
        elif Parch == 1:
            return 24
        elif Parch ==2:
            return 17
        elif Parch ==3:
            return 33
        elif Parch==4:
            return 45
        else:
            return 30                   #This value is the average age for the whole sample

    else:
        return Age

titanic_data['Age']= titanic_data[['Age', 'Parch',]].apply(age_approx, axis=1 ) #This function will replace mising values by
                                                                                ##the mean age of each Parch category (1 to 6)
titanic_data.isnull().sum()

titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop= True)            #This will reset the indexes for the rows.
print(titanic_data.info()) #Now there are no NAs


#Next step is to convert categorical variables into dummy ones

from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()

gender_cat=titanic_data['Sex']

gender_encoded= label_encoder.fit_transform(gender_cat)
gender_encoded[0:5]                                     #To check if things worked

#is 1 male or male?? Let's check it out
titanic_data.head()         #male= 1, female=0

gender_DF= pd.DataFrame(gender_encoded, columns=['male_gender']) #Transforming the male_gender array into DF

embarked_cat=titanic_data['Embarked']

embarked_encoded = label_encoder.fit_transform(embarked_cat)
embarked_encoded[0:100]

#Now, as we saw before, S was the first value for embarked in the first row and C was the second. This means
##S is equal to 2, C is equal to 0, and the other value will be 1. Our issue here is that we need binary values
### but embarked is a multinomial one.

#Therefore, we will create three different columns to make embarked binary through them

from sklearn.preprocessing import OneHotEncoder
binary_encoder= OneHotEncoder(categories='auto')

embarked_1hot= binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))
#The later created data is an array. We want it to be a matrix, so

embarked_1hot_mat = embarked_1hot.toarray()

#Then we transfor it into a PD DF with column named the place it refers to.
embarked_DF = pd.DataFrame(embarked_1hot_mat, columns=['C', 'Q', 'S'])
embarked_DF.head()

#We will drop the non-binary data now
titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

#Lastly, we concatenate the new variables into the DF

titanic_dummy= pd.concat([titanic_data, gender_DF, embarked_DF], axis=1, verify_integrity=True).astype(float)

titanic_dummy[0:5] #They're binary now



#We want to check heatmaps to see correlations

sb.heatmap(titanic_dummy.corr())

#LogReg assume variables are not correlated. We need to drop correlated variables.
##In the heatmap, we see correlation between pclass and fare, and pclass with age. We will drop the fare and pclass column


titanic_dummy.drop(['Fare'], axis=1, inplace=True)

#We will check if the predictor number, as LogReg requires 50 records per variable.

titanic_dummy.info()        #Which we do have.
titanic_dummy.isnull().sum() #We check one last time for NAs

#And we remove 'em

titanic_dummy= titanic_dummy.dropna()
#Next step is to break the data into test and train sets.

X_train, X_test, y_train, y_test = train_test_split(titanic_dummy.drop('Survived', axis=1), titanic_dummy['Survived'], test_size=0.2,
                                                    random_state= 200 )


print(X_train.shape)
print(y_train.shape)


X_train[0:5]



LogReg= LogisticRegression(solver='liblinear')

LogReg.fit(X_train, y_train)


y_pred=LogReg.predict(X_test)       #This will tell us how the model performed

print(classification_report(y_test, y_pred))


#To this data will be applied a k-fold cross validation and a confusion matrix.

y_train_pred= cross_val_predict(LogReg, X_train, y_train,  cv=5)

confusion_matrix(y_train, y_train_pred) #The Confusion matrix tells us how many predictions were true and how many were
                                        ##false. In this case, we've got 384 and 181 predictions correct and 56 and 90
                                        ###were false.

precision_score(y_train, y_train_pred)  #Our model is performing pretty well. true positives are more than the false positives.


#Next, a test prediction will take place. For this matter, we will take a random person and test if it fits.
titanic_dummy[863:864]
test_passenger = np.array([865, 2, 42, 0, 0, 0, 0, 0, 1]).reshape(1,-1)

print(LogReg.predict(test_passenger)) #The prediction is true.

print(LogReg.predict_proba(test_passenger)) #What this number tells us is that the probability of surviving is 0.74 with given
                                            ## variables.




                                                ####K-means####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import scipy as sp
import seaborn as sb


sales='C:/Users/CrsTn/Documents/GitHub/RoadToDS2/Used Car Sales Database/Salesbybrand.csv'

salesbybrand=pd.read_csv(sales)
#You will need to scale variables and take a look at the scatterplot or data table to estimate a number of centroids
##to apply in the k parameter value.
#first, I will check for NA's

salesbybrand.isnull().sum()         #No NAs

#Now we will check for data usage and transform variables into less memory consuming ones
salesbybrand.info() #Data usage is now over 5.4 MB

salesbybrand['engineSize']=salesbybrand['engineSize'].astype(np.float16)
salesbybrand['mpg']=salesbybrand['mpg'].astype(np.float16)

salesbybrand['tax']=salesbybrand['tax'].astype(np.int16)
salesbybrand['year']=salesbybrand['year'].astype(np.int16)

salesbybrand['mileage']=salesbybrand['mileage'].astype(np.int32)

salesbybrand.info() #Now it is 3.3 MB for this command. Pretty good anyways.

salesbybrand.head()

#Cleaning the DB from outliers


sb.boxplot(x=salesbybrand['price'])
sb.histplot(salesbybrand['price'])


sb.boxplot(x=salesbybrand['mileage'])
sb.histplot(salesbybrand['mileage'])

#We can see in both cases lots of outliers. Again, I'll use z scores to delete them with 1.5 and 3.0 STDs, just to see
##which one fits the model better.

z_scores= pd.DataFrame(abs(sp.stats.zscore(salesbybrand['price'])))
z_scores2=pd.DataFrame(abs(sp.stats.zscore(salesbybrand['mileage'])))


z_scores3=pd.concat([z_scores, z_scores2], axis=1)


NoOutliers_bybrand = salesbybrand[(z_scores3<3).all(axis=1)]

NoOutliers_bybrand2 = salesbybrand[(z_scores3<1.5).all(axis=1)]

print(len(NoOutliers_bybrand))
print(len(NoOutliers_bybrand2))


#Next step, to take only the columns with numeric values, because K-means does not work properly with dummies or categorical values.
NoOutliers_bybrand.head()                                   #      ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']

#These are the new DFs I will use for the K-means algorythm
k_meansBybrand=NoOutliers_bybrand[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']]
k_meansBybrand2=NoOutliers_bybrand2[['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']]
#The next step is to convert brands into dummies

k_meansBybrand2.corr()
                                                                             ###sb.heatmap(corrbmw, annot=True)
#Now, we scale the values.
finalsales=scale(k_meansBybrand)
finalsales2=scale(k_meansBybrand2)

#Finally... K-means. For the number of clusters, the elbow method will be applied.

from yellowbrick.cluster import KElbowVisualizer
model=KMeans()

visualizer= KElbowVisualizer(model, k=(1,15), timings=True)
visualizer.fit(finalsales)

visualizer.show()

visualizer.fit(finalsales2)

clustering= KMeans()


finals