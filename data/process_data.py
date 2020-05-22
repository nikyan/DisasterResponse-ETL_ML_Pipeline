import sys
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)


def load_data(messages_filepath, categories_filepath):
    """
    A function to load messages and categories data.
    
    INPUT:  
    messages_filepath - filepath of messages data
    categories_filepath - filepath of categories data

    OUTPUT: 
    df - a dataframe which has messages and categories data cancatenated 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    A function to clean df dataframe and make it ready for machine learning algorithm.
     - Expand 'Categories' column to 36 columns
     - Take first row of 36 new columns and rename the columns
     - Clean the 36 categories column to have only 1 and 0
     - Concatenate the 36 categories column with df and drop the original categories column
     - Remove duplicates
    
    INPUT:  
    df - a dataframe that contains messages and categories data

    OUTPUT: 
    df - a clean dataframe which is ready for machine learning
    
    
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
        
    # select the first row of the categories dataframe
    row = categories.loc[1]
        
    # extract a list of new column names for categories.
    category_colnames = list(row)
    final_col = []
    for col in category_colnames:
        temp = col[:-2]
        final_col.append(temp)
       
    # rename the columns of `categories`
    categories.columns = final_col
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop("categories", axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], sort=False, axis=1)
    
    # drop duplicates
    df = df[~df.duplicated(keep='first')]
    
    # remove rows with NaNs, Ignore 'original', 'genre' columns
    columns = df.drop(['original', 'genre'], axis=1).head(1)
    df = df.dropna(subset=columns.columns)
    
    return df

def save_data(df, database_filename):
    """
    A function to save df to a sqlite db
    
    INPUT:  
    df - dataframe with clean data that is saved to a sqlite db
    database_filename - name of database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Message', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()