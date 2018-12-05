import betterbnb.betterbnb.dataset.Data_Grabbing as dg
import pandas as pd
from betterbnb.betterbnb.utils.common import sys_home

df = pd.read_csv(sys_home() + '/Documents/detect_landmarks/src/betterbnb/data/listings.csv', error_bad_lines=False)
t = dg.cleanf(-20,-10,df)

print(t)
