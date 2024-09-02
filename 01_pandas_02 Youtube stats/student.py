import pandas as pd
import json

"""
    ASSIGNMENT 1 (STUDENT VERSION):
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
"""


def Q1():
    """
        1. How many rows are there in the GBvideos.csv after removing duplications?
        - To access 'GBvideos.csv', use the path '/data/GBvideos.csv'.
    """
    data_df = pd.read_csv('./data/USvideos.csv')
    data_df = data_df.drop_duplicates()

    return data_df.shape[0]


def Q2(vdo_df):
    '''
        2. How many VDO that have "dislikes" more than "likes"? Make sure that you count only unique title!
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    filtered_df = vdo_df[vdo_df['dislikes'] > vdo_df['likes']]
    unique_videos = filtered_df.drop_duplicates(subset='title')

    return unique_videos.shape[0]


def Q3(vdo_df):
    '''
        3. How many VDO that are trending on 22 Jan 2018 with comments more than 10,000 comments?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - The trending date of vdo_df is represented as 'YY.DD.MM'. For example, January 22, 2018, is represented as '18.22.01'.
    '''
    filtered_df = vdo_df[(vdo_df['trending_date'] == '18.22.01') & (vdo_df['comment_count'] > 10000)]
    return filtered_df.shape[0]


def Q4(vdo_df):
    '''
        4. Which trending date that has the minimum average number of comments per VDO?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
    '''
    avg_comments = vdo_df.groupby('trending_date')['comment_count'].mean()
    min_avg_comments_date = avg_comments.idxmin()
    return min_avg_comments_date


def Q5(vdo_df):
    '''
        5. Compare "Sports" and "Comedy", how many days that there are more total daily views of VDO in "Sports" category than in "Comedy" category?
            - GBvideos.csv has been loaded into memory and is ready to be utilized as vdo_df
            - The duplicate rows of vdo_df have been removed.
            - You must load the additional data from 'GB_category_id.json' into memory before executing any operations.
            - To access 'GB_category_id.json', use the path '/data/GB_category_id.json'.
    '''

    # Load category data
    with open('./data/US_category_id.json') as f:
        category_data = json.load(f)
    category_mapping = {int(item['id']): item['snippet']['title'] for item in category_data['items']}
    sports_ids = [key for key, value in category_mapping.items() if value == 'Sports']
    comedy_ids = [key for key, value in category_mapping.items() if value == 'Comedy']
    sports_comedy_df = vdo_df[vdo_df['category_id'].isin(sports_ids + comedy_ids)]
    grouped_df = sports_comedy_df.groupby(['trending_date', 'category_id'])['views'].sum().unstack(level='category_id')
    grouped_df.columns = ['Comedy_views' if col in comedy_ids else 'Sports_views' for col in grouped_df.columns]
    grouped_df = grouped_df.T.groupby(grouped_df.columns).sum().T
    comparison_df = grouped_df['Sports_views'] > grouped_df['Comedy_views']
    more_sports_days = comparison_df.sum()

    return more_sports_days


if __name__ == '__main__':
    data = pd.read_csv('./data/USvideos.csv')
    data = data.drop_duplicates()

    print(Q1())
    print(Q2(data))
    print(Q3(data))
    print(Q4(data))
    print(Q5(data))
