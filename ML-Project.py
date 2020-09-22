########################################################################################################################
################################################## FUNCTION DECLARATION ################################################
########################################################################################################################


################################
### VALUE COUNTING FUNCTIONS ###
################################


# Input Parameters:
#   -   df0 - [pandas DataFrame] - the original Dataframe
# Output Parameters:
#   -   y_values - [list] - Percentage of null values
#   -   labels - [list] - Name of columns
def nullcounter(df0):

    y_values = [] # list to hold x-values
    ticklabels = [] # list to hold y-values

    df_null0 = (df0.isnull().sum()/df.shape[0] * 100) # dataframe of null values as percentage

    y_values = df_null0.values.tolist()
    ticklabels = df_null0.index.tolist()


    # Loop to round off values to 3 decimal places
    i = 0
    for member in y_values:
        temp = round(member, 3)
        y_values[i] = temp
        i+=1

    return(y_values, ticklabels)


# FUNCTION TO DETERMINE FREQUENCY OF ALL VALUES IN A COLUMN
#
# Input Parameters:
#   -   df0 - [pandas DataFrame] - the original Dataframe
#   -   columnnmae0 - [string]  -   Name of the Column in which to count values
# Output Parameters:
#   -   y_values - [list] - Percentage of null values
#   -   labels - [list] - Name of columns
def valuecounter(df0, columnnmae0):

    df_valcount = (df0[columnnmae0].value_counts()/df0.shape[0] *100)

    y_values_temp = df_valcount.values.tolist()
    ticklabels_temp = df_valcount.index.tolist()

    l = len(ticklabels_temp)

    y_values = []
    ticklabels = []
    n = 0

    for i in range(0,l):
        j = y_values_temp[i]
        k = ticklabels_temp[i]

        if j >= 1:
            y_values.append(round(j, 3))
            ticklabels.append(ticklabels_temp[i])

        if j < 1:
            n = n +j

    if n > 0:
        y_values.append(round(n, 3))
        ticklabels.append("other")

    return(y_values, ticklabels)

# FUNCTION TO DETERMINE FREQUENCY OF ALL VALUES IN A SERIES
#
# Input Parameters:
#   -   df0 - [pandas DataFrame] - the series from which we need to count values
# Output Parameters:
#   -   y_values - [list] - Percentage of null values
#   -   labels - [list] - Name of columns
def valuecounter_series(df0):

    df_valcount = (df0.value_counts()/df0.shape[0] *100)

    y_values_temp = df_valcount.values.tolist()
    ticklabels_temp = df_valcount.index.tolist()

    l = len(ticklabels_temp)

    y_values = []
    ticklabels = []
    n = 0

    for i in range(0,l):
        j = y_values_temp[i]
        k = ticklabels_temp[i]

        if j >= 1:
            y_values.append(round(j, 3))
            ticklabels.append(ticklabels_temp[i])

        if j < 1:
            n = n +j

    if n > 0:
        y_values.append(round(n, 3))
        ticklabels.append("other")

    return(y_values, ticklabels)

# FUNCTION TO DETERMINE FREQUENCY OF ALL VALUES IN A COLUMN
#
# Input Parameters:
#   -   df0 - [pandas DataFrame] - the original Dataframe
# Output Parameters:
#   -   y_values - [list] - Percentage of null values
#   -   labels - [list] - Name of columns
def uniquecounter(df0):

    df_valcount = (df0.nunique())

    y_values = df_valcount.values.tolist()
    ticklabels = df_valcount.index.tolist()

    return(y_values, ticklabels)


####################################
### FUNCTOION TO DEAL WITH DATES ###
####################################


# FUNCTION TO PROCESS DATES #
#
# Input Parameters:
#   -   df0 - [panda's DataFrame]
#   -   columnname0 - [string] - Name of the column containing dates
# Output Parameters:
#   -   df_complainttime - [panda's DataFrame] - DataFrame containing Month, Date and Day of the week
def dateprocessing(df0, columnname0):
    timeofcomplaint = pd.to_datetime(df0[columnname0])

    day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    month = ["", "January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    timeofcomplaint_weekday = []
    timeofcomplaint_month = []
    timeofcomplaint_day = []

    for member in timeofcomplaint:
        timeofcomplaint_weekday.append(day_of_week[member.dayofweek])
        timeofcomplaint_month.append(month[member.month])
        timeofcomplaint_day.append(member.day)

    df_complainttime = pd.DataFrame({"month":timeofcomplaint_month, "date": timeofcomplaint_day, "weekday": timeofcomplaint_weekday})
    return(df_complainttime)


# FUNCTION TO ARRANGE WEEKDAYS #
#
#   Input Parametes:
#   -   y_values - [list] - list of days
#   -   dayeweek - [list] - list of number of complaints recieved on each weekday
#   Output Parameters
#   -   numdays - [list] - list of number of days
#   -   dayweek_ordered - [list] - list of days but arranged such that Monday is the first day
def weekproces(y_values, dayweek):
    dayweek_arranged = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    dayweek_ordered = []
    numdays = []
    for member in dayweek_arranged:
        for i in range(0,7):
            if dayweek[i] == member:
                dayweek_ordered.append(dayweek[i])
                numdays.append(y_values[i])

    return(numdays, dayweek_ordered)


# FUNCTION TO ARRANGE MONTH #
#
#   Input Parametes:
#   -   y_values - [list] - list of days
#   -   monthyear - [list] - list of number of complaints recieved in each month
#   Output Parameters
#   -   numdays - [list] - list of number of days
#   -   dayweek_ordered - [list] - list of months but arranged such that Monday is the first day
def monthprocess(y_values, monthyear):
    monthyear_arranged = ["January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    monthyear_ordered = []
    numdays = []
    for member in monthyear_arranged:
        for i in range(0,12):
            if monthyear[i] == member:
                monthyear_ordered.append(monthyear[i])
                numdays.append(y_values[i])

    return(numdays, monthyear_ordered)


###################################
### FUNCTIONS TO GENERATE PLOTS ###
###################################


### FUNCIONS TO GENERATE PLOTS ###

# FUNCTION TO GENERATE HORIZONTAL BAR PLOTS
#
# Input Arguments
#   -   y_values - [List] - Values for Y-axis
#   -   ticklabels_x - [list] - x-tick labels
#   -   label_x - [list] - X-axis Labels
#   -   label_y - [list] - Y-axis Labels
#   -   filename_plt - [string] - Name of the plot
#   -   refrence_line - [integer] - integer value to draw Refrence Line
# Output Arguments
#   -   Horizontal Bar plot
def plot_bar(y_values, ticklabels_x, label_x, label_y, ttl, filename_plt, refrence_line):
    xpos = np.arange(len(y_values))  # x-positions

    fig, ax = plt.subplots()

    ax.bar(xpos, y_values, width=0.5) # plotting Bar Graphs
    if refrence_line > 0: # Condition to draw refrence line
        ax.axhline(refrence_line, color='k', linewidth=0.3, linestyle='dashed')

    # x-axis operations
    plt.xticks(fontsize=7)  # controlling the foent size of x-tick labells
    ax.set_xticks(xpos)  # setting the position of x-tick labells
    ax.set_xticklabels(ticklabels_x, rotation=90, ha='center')  # setting up x-tick labells
    ax.set_xlabel(label_x, fontsize=10)  # setting up x-axis labell

    # y-axis operations
    plt.yticks(fontsize=7)  # controlling the foent size of x-tick labells
    ax.set_ylabel(label_y, fontsize=10)  # setting up y-tick labells

    # Eliminating the spine
    ax.spines['right'].set_visible(False)  # eliminating right border
    ax.spines['top'].set_visible(False)  # eliminating top border

    # Setting up annotations
    lmt_l, lmt_u = ax.get_ylim()  # y-axis limits
    for i in range(0, len(xpos)):
        plt.text(x=xpos[i] - 0.125, y=y_values[i] + (0.01 * lmt_u), s=y_values[i], fontsize=7, rotation=90)

    plt.title(ttl, y=1.1)

    # For plot zooming
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.savefig(filename_plt, bbox_inches='tight')  # saving the plot
    plt.close(fig) # Closing the window after plotting


# FUNCTION TO GENERATE HORIZONTAL BAR PLOTS
#
# Input Arguments
#   -   y1_values, y2_values - [List] - Values for Y-axis(two lists)
#   -   group_plt - [list] - Name of groups
#   -   label_x - [list] - X-axis Labels
#   -   label_y - [list] - Y-axis Labels
#   -   xticklabels_plt - [list] - X-axis Tick Labels
#   -   filename_plt - [string] - Name of the plot
# Output Arguments
#   -   Grouped Horizontal Bar plot
def plot_bar_grouped(y1_values, y2_values, group_plt, label_x, label_y, xticklabels_plt, filename_plt):
    ofst = 0.35  # offset
    xpos = np.arange(len(y1_values))  # x-positions

    fig, ax = plt.subplots()

    grp1 = ax.bar(xpos - ofst / 2, y1_values, ofst, label=group_plt[0])
    grp2 = ax.bar(xpos + ofst / 2, y2_values, ofst, label=group_plt[1])

    # x-axis operations
    plt.xticks(fontsize=7)  # controlling the font size of x-tick labells
    ax.set_xticks(xpos)  # setting the position of x-tick labells
    ax.set_xticklabels(xticklabels_plt, rotation=0, ha='center')  # setting up x-tick labells
    ax.set_xlabel(label_x, fontsize=10)  # setting up x-axis labell

    # y-axis operations
    plt.yticks(fontsize=7)  # controlling the foent size of x-tick labells
    ax.set_ylabel(label_y, fontsize=10)  # setting up y-tick labells

    # Eliminating the spine
    ax.spines['right'].set_visible(False)  # eliminating right border
    ax.spines['top'].set_visible(False)  # eliminating top border

    # Setting up annotations
    lmt_l, lmt_u = ax.get_ylim()  # y-axis limits
    for i in range(0, len(xpos)):
        plt.text(x=xpos[i] - ofst / 1.5, y=y1_values[i] + (0.01 * lmt_u), s=y1_values[i], fontsize=7, rotation=90)
        plt.text(x=xpos[i] + ofst / 2.75, y=y2_values[i] + (0.01 * lmt_u), s=y2_values[i], fontsize=7, rotation=90)

    ax.legend(fontsize=7, ncol=2, loc='center', bbox_to_anchor=(0.5, 1.15), labelspacing=1)  # setting Legend

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plt.savefig(filename_plt, bbox_inches='tight')  # saving the plot
    plt.close(fig) # Closing the window after plotting


########################################################################################################################
################################################ MAIN BODY OF THE PROGRAM ##############################################
########################################################################################################################

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Reading the DataFrame
df = pd.read_csv("Edureka_Consumer_Complaints_train.csv")

# Dropping Un-necessary Columns
df = df.drop(["Complaint ID"], axis=1)

################
################
#### TASK-1 ####
################
################

print("STARTING TASK - 1")
print("___________________________________________________________________________________________________________________________________")
now = datetime.now()
print("Mandatory EDA initiated at ", now)
print("\n")

# Plotting Null Value Count
y_values_P1, ticklabels_x_P1 = nullcounter(df)
title_P1 = "Columns with percentage of missing values"
plot_bar(y_values_P1, ticklabels_x_P1, "Columns", "Null Values Percentage", title_P1,"task1_plot_1.pdf", 10)
# Note: Only the Columns which have more than 10% missing values will be dropped

# Dropping Columns
df = df.drop(["Sub-product", "Sub-issue", "Consumer consent provided?", "ZIP code", "Tags"], axis=1)
print("Dropping non-essetial columns with more than 10 percent missing values")
print("The Following Columns are dropped:")
print("1: Sub-Product")
print("2: Sub-issue")
print("3:Consumer consent provided?")
print("4:ZIP code")
print("5:Tags")
print("\n")

# Note: Only the columns which have more than 10% missing values and my not contribute to Model are dropped
# Note: Even though ZIP code has less than 10% missing values and can also provide insight into geographical Skew
#       But the column is still dropped because some entries do not display the full ZIP code and some digits are hidden.

# Unique Values in each columns
y_values_P2, ticklabels_x_P2 = uniquecounter(df)
title_P2 = "Number of Unique Values in each column"
plot_bar(y_values_P2, ticklabels_x_P2, "Column", "Unique Values", title_P2,"task1_plot_2.pdf", 0)
print("Note*: Visualization with number of unique values in each column is saved as bar graph(task1_plot_2.pdf)")
print("\n")

# Value count of Categorical Columns (Issue)
y_values_P3, ticklabels_x_P3 = valuecounter(df, "Issue")
title_P3 = "Top Issues raised by Consumers"
plot_bar(y_values_P3, ticklabels_x_P3, "Issue", "Number of Complaints", title_P3, "task1_plot_3.pdf", 0)
print("Loan modification, collection and foreclosure - Top Issue raised by consumers")
print("\n")

# Value count of Categorical Columns (Products)
y_values_P4, ticklabels_x_P4 = valuecounter(df, "Product")
title_P4 = "Products which recieved the most complaints"
plot_bar(y_values_P4, ticklabels_x_P4, "Products", "Number of Complaints", title_P4,"task1_plot_4.pdf", 0)
print("More than 50 percent complaints are regarding the following products:")
print("1: Mortgage = 32.626")
print("2: Debt Coolection = 18.125")
print("\n")

# Value count of Categorical Columns (Company)
y_values_P5, ticklabels_x_P5 = valuecounter(df, "Company")
title_P5 = "Number of Complaints recieved by each company"
plot_bar(y_values_P5, ticklabels_x_P5, "Company", "Number of Complaints", title_P5, "task1_plot_5.pdf", 0)
print("Bank of america has the largest share of complaint i.e. 9.727 percent")
print("\n")
# Note: It can be observed other companies(companies with less than 1% complaints) have the maximum Number of complaints.
#       But "Bank of America" is the single company with highest number of individual complaints

# Value count of Categorical Columns (Submitted via)
y_values_P6, ticklabels_x_P6 = valuecounter(df, "Submitted via")
title_P6 = "Number of Complaints recieved via each medium"
plot_bar(y_values_P6, ticklabels_x_P6, "Mode of Submission", "Number of Complaints", title_P6,"task1_plot_7.pdf", 0)
print("Highest percentage of complaints are submitted via Web")
print("\n")

# Value count of Categorical Columns (State)
y_values_P7, ticklabels_x_P7 = valuecounter(df, "State")
title_P7 = "Geographical Distribution of Complaints"
plot_bar(y_values_P7, ticklabels_x_P7, "State", "Number of Complaints", title_P7,"task1_plot_6.pdf", 0)
print("Maximum number of complaints origintate from the state of California(14.595 percent)")
print("\n")

# Value count of Categorical Columns (Submitted via)
y_values_P8, ticklabels_x_P8 = valuecounter(df, "Company response to consumer")
title_P8 = "Most Common Response by Companies"
plot_bar(y_values_P8, ticklabels_x_P8, "Company Response", "Number of Complaints", title_P8, "task1_plot_8.pdf", 0)
print("Closed with explanation is the most common response by the company")
print("\n")

# Value count of Categorical Column (Date Recieved) - Date
df_complaintdates = dateprocessing(df, "Date received")
y_values_P9, ticklabels_x_P9 = valuecounter(df_complaintdates, "date")
df_temp = pd.DataFrame({"dayofmonth": ticklabels_x_P9, "numberofcomplaints": y_values_P9}) # Data Frame for arranging values
df_temp = df_temp.sort_values("dayofmonth", ascending=True) # sorting values by day of the month i.e. Date
y_values_P9 = df_temp["numberofcomplaints"].values.tolist() # Creating a list of dates
ticklabels_x_P9 = df_temp["dayofmonth"].values.tolist() # Creating a  list of number of days
title_P9 = "Number of Complaints Recieved on each day of the Month"
plot_bar(y_values_P9, ticklabels_x_P9, "Date", "Number of Complaints", title_P9, "task1_plot_9.pdf", 0)
print("The number of complaints recieved on 31 day of the month is significantly less that other days")
print("\n")

# Value count of Categorical Column (Date Recieved) - Month
df_complaintdates = dateprocessing(df, "Date received")
y_values_temp, ticklabels_x_temp = valuecounter(df_complaintdates, "month")
y_values_P10, ticklabels_x_P10 = monthprocess(y_values_temp, ticklabels_x_temp)
title_P10 = "Number of Complaints recieved each Month "
plot_bar(y_values_P10, ticklabels_x_P10, "Month", "Number of Complaints", title_P10,"task1_plot_10.pdf", 0)
print("Number of complaints recieved in the months of March to June are significantly higher than rest of the months")
print("\n")

# Value count of Categorical Column (Date Recieved) - Day of the week
df_complaintdates = dateprocessing(df, "Date received")
y_values_temp, ticklabels_x_temp = valuecounter(df_complaintdates, "weekday")
y_values_P11, ticklabels_x_P11 = weekproces(y_values_temp, ticklabels_x_temp)
title_P11 = "Number of Complaints recieved on each day of the week"
plot_bar(y_values_P11, ticklabels_x_P11, "Day of the Week", "Number of Complaints", title_P11,"task1_plot_11.pdf", 0)
print("Number of complaints recived on weekdays are almost equal but significantly less number of complaints are recieved on weekends")
print("\n")


# Effect of Timely/Late Response on Consumer Disputed
df_temp = df[["Consumer disputed?", "Timely response?"]]
df_timely = df_temp[df_temp["Timely response?"]=="Yes"]
df_late = df_temp[df_temp["Timely response?"]=="No"]
y_values_temp1, x_values_P12 = valuecounter(df_timely, "Consumer disputed?")
y_values_temp2, x_values_P12 = valuecounter(df_late, "Consumer disputed?")
y_values_1P12 = [y_values_temp1[0], y_values_temp2[0]]
y_values_2P12 = [y_values_temp1[1], y_values_temp2[1]]
plot_bar_grouped(y_values_1P12, y_values_2P12, x_values_P12, "Response", "Number of Complaints", ["Timely Response", "Late Response"], "task1_plot12.pdf")
print("Timely/Late Response has no effect on weather or not the consumer will dispute the decision")
print("\n")

now = datetime.now()
print("TASk 1 COMPLETED AT ", now)
print("___________________________________________________________________________________________________________________________________")
print("\n")

############################################################################################################################################

################
################
#### TASK-2 ####
################
################

now = datetime.now()
print("TASK-2 STARTED AT ", now)
print("\n")

# DataFrame Operation
df_nlp = pd.read_csv("Edureka_Consumer_Complaints_train.csv", usecols=["Consumer complaint narrative", "Product"]) # Reading DataFrame
print("The NLP DataFrame has ", df_nlp.shape[0], "rows and", df_nlp.shape[1]," columns")
df_nlp.dropna(inplace=True) # Dropping Null Values
print("The NLP DataFrame has ", df_nlp.shape[0], " rows and", df_nlp.shape[1]," columns after dropping NA values")

# Generating Independent and Dependent Variables
X_train_nlp = df_nlp["Consumer complaint narrative"]
Y_train_nlp = df_nlp["Product"]

# Visualizing balance of Products in the training dataset
y_values_P1, ticklabels_x_P1 = valuecounter_series(Y_train_nlp)
title_P1 = "Balance of Target Classes for NLP DataFrame"
plot_bar(y_values_P1, ticklabels_x_P1, "Product", "Count", title_P1,"task2_plot_1.pdf", 0)
print("Note*: The Dataset is not balanced, appropiate steps need to be taken while using Machine Learning Algorithms")
print("\n")

# Using TfIdf Vectorizer
# vectorizer_TfIdf = TfidfVectorizer(stop_words="english")
# X_train_count = vectorizer_TfIdf.fit_transform(X_train_nlp)

# print(vectorizer_TfIdf.get_feature_names())

vectorizer_count = CountVectorizer(stop_words="english")
X_train_count = vectorizer_count.fit_transform(X_train_nlp)


### MODELS ###
regression_logistic = LogisticRegression(multi_class="multinomial", class_weight="balanced") #Logistic REgression (Multinomial)
classifier_DT = DecisionTreeClassifier(class_weight="balanced") # Decision Tree Classifier
classifier_RF = RandomForestClassifier(random_state=0, class_weight="balanced") # Random Forest Classifier
model_multiNB = MultinomialNB() # Multinomial Naive Bayes
model_linearsvc = LinearSVC() # Linear SVC


# ### PARAMETER GRIDS ###
# # Logistic Regression (Multinomial)
# param_regression_logistic = {"solver": ["lbfgs", "newton-cg", "sag"],
#                              "C": [0.5, 1, 1.5, 2],
#                              "max_iter": [10, 50, 100, 250, 500, 750, 1000]}
# #_____________________________________________________________________________________________________________#
#
# # Decision Tree Classifier #
# param_classifier_DT = {"criterion": ["gini", "entropy"],
#                        "splitter": ["best", "random"]}
# #_____________________________________________________________________________________________________________#
#
#
# # Random Forest Classifier
# param_classifier_RF = {"n_estimators": [100, 200],
#                        "criterion": ["gini", "entropy"]}
# #_____________________________________________________________________________________________________________#
#
# # Multinomail Naive Bayes
# param_model_multiNB = {"alpha": [0.0125, 0.025, 0.05, 0.1]}
# #_____________________________________________________________________________________________________________#

# LinearSVC #
param_model_linearsvc = {"penalty": ["l1", "l2"],
                          "loss": ["hinge", "squared_hinge"],
                          "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}
#_____________________________________________________________________________________________________________#


# #### APPLYING GRIDSEARCH CROSS VALIDATION ###
#
# # Logistic Regression(Multinomial)
# now = datetime.now()
# print("[1 of 5] - Hyper Parameter Tuning For Logistic Regreesion  initiated at", now)
#
# model_gscv = GridSearchCV(estimator=regression_logistic, param_grid=param_regression_logistic, cv=5)
# model_gscv.fit(X_train_count, Y_train_nlp)
# df_gscv_results = pd.DataFrame(model_gscv.cv_results_)
# df_gscv_results.to_csv("logistic_regression_results_nlp_count.csv", index=False)
#
# now = datetime.now()
# print("Hyper Parameter Tuning For Logistic Regreesion  completed at", now)
# #_____________________________________________________________________________________________________________#
#
# # Decision Tree Classifer #
# now = datetime.now()
# print("[2 of 5] - Hyper Parameter Tuning For Decision Tree initiated at", now)
#
# model_gscv = GridSearchCV(estimator=classifier_DT, param_grid=param_classifier_DT, cv=5)
# model_gscv.fit(X_train_count, Y_train_nlp)
# df_gscv_results = pd.DataFrame(model_gscv.cv_results_)
# df_gscv_results.to_csv("descisiontree_results_nlp_count.csv", index=False)
#
# now = datetime.now()
# print("Hyper Parameter Tuning For Decision Tree completed at", now)
# #_____________________________________________________________________________________________________________#
#
# # Random Forest Classifier
# now = datetime.now()
# print("[3 of 5] - Hyper Parameter Tuning for Random Forest initiated at", now)
#
# model_gscv = GridSearchCV(estimator=classifier_RF, param_grid=param_classifier_RF, cv=5)
# model_gscv.fit(X_train_count, Y_train_nlp)
# df_gscv_results = pd.DataFrame(model_gscv.cv_results_)
# df_gscv_results.to_csv("randomforest_results_nlp_count.csv", index=False)
#
# now = datetime.now()
# print("[3 of 5] - Hyper Parameter Tuning for Random Forest completed at", now)
# #_____________________________________________________________________________________________________________#
#
# # Multinomial Naive Bayes
# now = datetime.now()
# print("[4 of 5] - Hyper Parameter Tuning for Multinomial NB initiated at", now)
#
# model_gscv = GridSearchCV(estimator=model_multiNB , param_grid=param_model_multiNB, cv=5)
# model_gscv.fit(X_train_count, Y_train_nlp)
# df_gscv_results = pd.DataFrame(model_gscv.cv_results_)
# df_gscv_results.to_csv("multinomialNB_results_nlp_count_count.csv", index=False)
#
# now = datetime.now()
# print("Hyper Parameter Tuning for Multinomial NB completed at", now)
# #_____________________________________________________________________________________________________________#

# LinearSVC #
now = datetime.now()
print("[5 of 5] - Hyper Parameter Tuning for LinearSVC initiated at", now)

model_gscv = GridSearchCV(estimator=model_linearsvc, param_grid=param_model_linearsvc, cv=5)
model_gscv.fit(X_train_count, Y_train_nlp)
df_gscv_results = pd.DataFrame(model_gscv.cv_results_)
df_gscv_results.to_csv("linearsvc_results_nlp_count.csv", index=False)

now = datetime.now()
print("Hyper Parameter Tuning for LinearSVC completed at", now)
#_____________________________________________________________________________________________________________#


print("Skipping Hyper-Parameter Tuning, Selecting Best model and parameters from earlier results")
print("\n")
print("Using LinearSVC to train for NLP predictions")
print("\n")
model_nlp = LinearSVC(loss="hinge", penalty="l2", tol=0.1,)
model_nlp.fit(X_train_count, Y_train_nlp)

now = datetime.now()
print("training for NLP completed at", now)
print("________________________________________________________________________________________________________")


# #####################
# #####################
# #### PREDICTIONS ####
# #####################
# #####################
#
#
# ### PREDICTIONS FOR NLP ###
#
# print("________________________________________________________________________________________________________")
# now = datetime.now()
# print("NLP PREDICTION STARTED AT", now)
# print("\n")
#
# df_nlp_test = pd.read_csv("Edureka_Consumer_Complaints_train.csv", usecols=["Consumer complaint narrative", "Product"])
#
# # Dropping Null Values
# df_nlp_test.dropna(inplace=True)
# print("The final test dataframe has ", df_nlp_test.shape[0], "rows", df_nlp_test.shape[1], "columns")
#
# X_test_nlp = df_nlp_test["Consumer complaint narrative"]
# Y_test_nlp = df_nlp_test["Product"]
#
# # Using TfIdf Vectorizer
# vectorizer_TfIdf = TfidfVectorizer(stop_words="english")
# X_test_idf = vectorizer_TfIdf.fit_transform(X_test_nlp)
#
# # Predicting Results
# Y_pred_nlp_test = model_nlp.predict(X_test_idf)
#
#
# Z = pd.Series(Y_pred_nlp_test)
# Z.to_csv("nlp_pred.csv", index=False)
#
# now = datetime.now()
# print("NLP PREDICTION COMPLETED AT", now)
# print("________________________________________________________________________________________________________")
# print("________________________________________________________________________________________________________")
# print("\n")
