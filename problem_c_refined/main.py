import pandas as pd
from utils import *

medal = pd.read_csv('./data/summerOly_medal_counts.csv', encoding='utf-8')
athlete = pd.read_csv('./data/summerOly_athletes.csv', encoding='utf-8')
hosts = pd.read_csv('./data/summerOly_hosts.csv')
program = pd.read_csv('./data/summerOly_programs.csv', encoding='windows 1258')

medal['NOC'] = medal['NOC'].apply(normalize_text).apply(clean_text)

medal = medal.groupby('NOC').filter(lambda x: x['Year'].max() >= 2024)

medal_grouped = medal.groupby('NOC')
print(medal_grouped.size())
athlete_grouped = athlete.groupby('NOC')
print(athlete_grouped.size())

time = 5
medal_class_2 = medal[medal['Year'] >= 2024 - time*4].groupby('NOC').filter(lambda x: len(x) >= time)
stastics = medal_class_2[['NOC','Rank','Gold', 'Silver', 'Bronze', 'Total']].groupby('NOC').agg(['mean','std'])
