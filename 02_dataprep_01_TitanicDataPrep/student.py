import pandas as pd
from sklearn.model_selection import train_test_split

"""
    ASSIGNMENT 2 (STUDENT VERSION):
    Using pandas to explore Titanic data from Kaggle (titanic.csv) and answer the questions.
"""


def Q1(df):
    """
        Problem 1:
            How many rows are there in the “titanic.csv?
            Hint: In this function, you must load your data into memory before executing any operations. To access titanic.csv, use the path /data/titanic.csv.
    """
    #data_df = pd.read_csv('/data/titanic.csv')
    return df.shape[0]


def Q2(df):
    '''
        Problem 2:
            Drop unqualified variables
            Drop variables with missing > 50%
            Drop categorical variables with flat values > 70% (variables with the same value in the same column)
            How many columns do we have left?
    '''
    thresh_value = int(df.shape[0] * 0.5)
    df.dropna(thresh=thresh_value, axis=1, inplace=True)


    for col in df.select_dtypes(include=['object']).columns:
      top_freq = df[col].value_counts(normalize=True).max()
      if top_freq > 0.7:
        df.drop(columns=[col], inplace=True)

    return df.shape[1]


def Q3(df):
    '''
       Problem 3:
            Remove all rows with missing targets (the variable "Survived")
            How many rows do we have left?
    '''
    df.dropna(subset=['Survived'], inplace=True)
    return df.shape[0]


def Q4(df):
    '''
       Problem 4:
            Handle outliers
            For the variable “Fare”, replace outlier values with the boundary values
            If value < (Q1 - 1.5IQR), replace with (Q1 - 1.5IQR)
            If value > (Q3 + 1.5IQR), replace with (Q3 + 1.5IQR)
            What is the mean of “Fare” after replacing the outliers (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['Fare'] = df['Fare'].apply(lambda fare: lower_bound if fare < lower_bound else upper_bound if fare > upper_bound else fare)

    mean_fare = round(df['Fare'].mean(), 2)
    return mean_fare


def Q5(df):
    '''
       Problem 5:
            Impute missing value
            For number type column, impute missing values with mean
            What is the average (mean) of “Age” after imputing the missing values (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    mean_age = df['Age'].mean()
    df['Age'] = df['Age'].fillna(mean_age)
    mean_age_after_impute = round(df['Age'].mean(), 2)
    return mean_age_after_impute


def Q6(df):
    '''
        Problem 6:
            Convert categorical to numeric values
            For the variable “Embarked”, perform the dummy coding.
            What is the average (mean) of “Embarked_Q” after performing dummy coding (round 2 decimal points)?
            Hint: Use function round(_, 2)
    '''
    df_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    mean_embarked_q = round(df_dummies['Embarked_Q'].mean(), 2)
    return mean_embarked_q


def Q7(df):
    '''
        Problem 7:
            Split train/test split with stratification using 70%:30% and random seed with 123
            Show a proportion between survived (1) and died (0) in all data sets (total data, train, test)
            What is the proportion of survivors (survived = 1) in the training data (round 2 decimal points)?
            Hint: Use function round(_, 2), and train_test_split() from sklearn.model_selection
    '''
    X = df.drop(columns='Survived')
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

    proportion_survived_train = y_train.value_counts(normalize=True)[1]
    proportion_survived_train_rounded = round(proportion_survived_train, 2)
    return proportion_survived_train_rounded
