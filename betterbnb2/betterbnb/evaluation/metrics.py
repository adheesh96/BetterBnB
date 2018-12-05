import pandas as pd
import gmplot
from IPython.display import display

raw_data = pd.read_csv('Users/lila/PycharmProjects/project2/betterbnb2/data/listings.csv')
print(raw_data.head(n=100))
display(raw_data.info())
