import pandas as pd
import matplotlib.pyplot as plt

def remove_bad_samples(df):
    # check for nan samples (i.e., invalid)
    return df


def cleanf(x,y,dl):
    """
    Grabs data
    :param x:
    :param y:
    :param dl:
    :return:
    """
    t = dl[dl.columns[x:y]]
    print(t)
    return t

# """cleans data"""
# def shapeofdf(df):
#     return df.shape

def hist1col(nameofcol, df):
    """defines shape of dataframe"""
    return df.hist(column= nameofcol)

"""makes a histogram of one column"""

def boxplotdf(df, title='', xlabel='',ylabel=''):
    df.boxplot(rot=45)
    plt.show()

"""generates box plot of data points"""

def drop_colum(df, nameofcol):
    return df.drop(nameofcol, axis=1)

    """delets single columns"""

def drop_lst_colum(lst, df):
    df.drop(lst, inplace=True)
    return df
"""deletes list of columns columns"""


def idxline(nameofindex):
    """
        this grabs the horizontal drop by the name of index

    :param nameofindex:
    :return:
    """

    return df.loc['nameofindex']

if __name__ == '__main__':

    df = pd.read_csv('/Users/lila/PycharmProjects/project2/betterbnb2/data/listings.csv', error_bad_lines=False)
    print(df.columns.values.tolist())



