'''Video no - 8, 11.4.21'''

######## performing EDA on train data  ########

import pandas as pd
bigmart = pd.read_csv("C:/Users/user/Desktop/ml/data sheet/Train.csv")

bigmart.columns

bigmart.drop('Item_Identifier',axis=1,inplace=True) #droping the column

### axis means column. here we have to choose what we have to drop ie column or row ie index
### inplace = true means If True, do operation inplace and return None. 

#handling missing value
bigmart.isnull().sum() #shows the na value ie not available value in the data
pd.isnull(bigmart) #it shows the na value in true false from which is difficult to understand
pd.isnull(bigmart).sum() # it shows the no of na value

bigmart.info()  # it shows the available value
'''above 4 are shows the null value in different format'''

#dropping the na value row
bigmart_nona = bigmart.dropna()

bigmart_nona.isnull().sum() # checking any other na value is there or not


###### imputation (imputation is done for the column in which na value is there and done individually for every column)
"""to know which one to use from mean/median/mode 
we have to find the outlier for that we have to do box plot"""

'''imputation for Item_Weight column'''

#to find the outlier use box plot
import matplotlib.pyplot as plt
plt.boxplot(bigmart_nona.Item_Weight) #here no outlier is there so we use mean for imputation

#here we use bigmart_nona because in bigmart data there are na values
# in box plot the middle line is the mean value
# to draw box plot give discrete data in x axis and in y axis give continous data

#using mean for imputation
bigmart.Item_Weight.fillna(bigmart_nona.Item_Weight.mean(),inplace = True)

"""here we use   bigmart_nona.Item_Weight.mean()   to replace the na values of Item_Weight
ie its the mean of the bigmart_nana.item_weight from the box plot. we put it directly by
bigmart_nona.Item_Weight.mean()  and replacing it with the na value by inplace = True"""

bigmart.isnull().sum() # to check whether there is null value is there or not after imputation


'''imputation for Outlet_Size column''' #imputation with mode because values are string

print(bigmart["Outlet_Size"]) #shows the values in it in the first and last 5 rows
bigmart.Outlet_Size.unique()  #shows the unique values
bigmart.Outlet_Size.value_counts()
#it shows the no of each unique value by which we can find the mode
#the mode is max no of unique value ie here the 'Medium'
#here we can replace the na values by the Medium but it will make the data wrong so 
#we will replace by a name 'others'

bigmart.Outlet_Size.fillna("others",inplace=True) #imputing na values with others
bigmart.Outlet_Size.value_counts() # to see na value replaced or not


###### FEATURE ENGINEERING   ########

'''we have to check for every column unique values. there are some data entry error like 
same thing in abbribation or in small lettor or capital letter. for which it shows no of
unique values for same thing . we have to make it one by replace() '''

# analysing the Item_Fat_content column

bigmart.Item_Fat_Content.unique() #finding unique values
bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace("low fat","LF") # replacing values
bigmart.Item_Fat_Content = bigmart.Item_Fat_Content.str.replace("LF","Low Fat").replace("reg","Regular") # replacing values and we can do it in a single line as this
bigmart.Item_Fat_Content.unique()  # checking
bigmart.Item_Fat_Content.value_counts() # checking 

# analysing the Item_Visibility column

"""in this column item visibility meanse probability of visibility of the item in bigmart
but 0 item visibiity is not possible so we replace the zero with madian"""
bigmart.Item_Visibility.describe()
bigmart.Item_Visibility = bigmart.Item_Visibility.replace(0,bigmart.Item_Visibility.median())
# replacing 0 with madian and saving it in the same variable because inplace() not work here
#### here we r using median because to how to use it .previously we used mean


# analysing the Item_Type column
bigmart.Item_Type.unique()
bigmart.Item_Type.value_counts()
# all values are ok , if we want cvhange anything we can use replace()

## analysing the Item_MRP column
# in it no na value we already checked 
plt.boxplot(bigmart.Item_MRP) # no out layer found ###good to go

## analysing the Outlet_Identifier column
#we dont need this column for prediction so we drop the column
bigmart.drop('Outlet_Identifier',axis=1,inplace=True)


## analysing the Outlet_Establishment_Year column
# here no na values found  so good to go

## analysing the Outlet_Location_Type column
bigmart.Outlet_Location_Type.unique()
# here no repetetion ie unique values found  so good to go

## analysing the Outlet_Type column
bigmart.Outlet_Type.unique()
# here no repetetion ie unique values found  so good to go

## analysing the Item_Outlet_Sales column
# here no na values found  so good to go



 