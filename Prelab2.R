rm(list = ls())
######### Lab Objectives ###########
# 1. Import datasets into RStudio and examine their contents.
# 2. Use functions on variables in a dataset.
# 3. Create subsets of data using logical indexing.


###### set the working directory 
getwd() # see the currect working directory
setwd("D:/2018 Fall/Biostats/Lab")


###### read in the data 
mydata <- read.csv("Handout2_CO2.csv")
# or
mydata <- read.csv("D:/2018 Fall/Biostats/Lab/Handout2_CO2.csv")

# Either way, you should now see mydata in the Environment window, 
# with the number of observations and variables it contains.
# The dataset becomes an object called a data frame. 
# A data frame consists of rows (observations) and columns (variables).


# Get a view of our dataframe
head(mydata) # or specify number of rows

# To call out a specific variable, use the data frame name 
# followed by $ and then the variable name. Eg:
mean(mydata$uptake)

# Check what values a categorical variable can be
levels(mydata$treatment)
# or you can display the Structure of our dataframe
str(mydata)


##### We can make frequency and contingency tables by using the function: table(). 
# Frequency table for treatment
table(mydata$location) 
# Contingency (or two-way) table by including two categoical variables
table(mydata$treatment,mydata$location)

# we can see that there are two treatment groups of equal number (42 plants each), 
# and also it has the same number of each group in the two locations.


##### Indexing a Data Frame
# we will use logical indexing to subset our data
# since dataframe has both rows and columns, so we can specify two indices
mydata[5,4]
 # leave one of the indices blank and view a specific row or column
mydata[8, ]
mydata[, 4]

# If we want to analyze the plants from the Quebec location only. 
# We can assign those plants to a separate data frame using logical indexing.
# We will create a new data frame, quebec, 
# which includes all of the observations (rows) from mydata 
# where the location variable is equal to ¡°Quebec¡±:

quebec <- mydata[mydata$location=="Quebec", ]
mydata$location=="Quebec"

# we can combines multiple logical expressions to 
# make a new data frame for the plants from Mississippi with 
# concentration values equal to 95
mississipi250 <- mydata[mydata$location == "Mississippi" & mydata$conc == 95, ]
print(mississipi250)


### Question 1:  How many observations are there 
# in the "chilled" treatment group for plant type "B"?
chilled_B <- mydata[mydata$treatment == "chilled" & mydata$plant_type == "B", ]
#  or
sum(mydata$treatment == "chilled" & mydata$plant_type == "B")

### Question 2: How many observations are there for plant type ¡°B¡± 
# that have a CO2 uptake of more than 40 umol/m2sec? 
sum(mydata$plant_type == "B" & mydata$uptake > 40)

### Question 3: What is the mean CO2 uptake for all the plant type ¡°B¡± observations?  
# Round to 1 decimal place and do not include units.
uptake_B <- mydata[mydata$plant_type == "B", 5]
round(mean(uptake_B),1)


##### Indexing a variable 
# Let's look at observations of a specific variable
# View uptake of plants with concentration of 500
mydata$conc
mydata$uptake[mydata$conc == 500]


# Find mean uptake of Quebec plants
uptake_QB <- mydata$uptake[mydata$location=='Quebec']
var(uptake_QB)

# Remember that do not inlcude comma when indexing a vector,
# do not forget to include the comma when indexing a data frame


