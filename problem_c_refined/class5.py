import pandas as pd
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import os

atheletes = pd.read_csv('./data/summerOly_athletes.csv',encoding='utf-8')
atheletes = atheletes.groupby('NOC').filter(lambda x: x['Year'].max() >= 2024)
medal_counts = atheletes.groupby(['NOC','Medal']).size().unstack(fill_value=0)

medal_counts['Total'] = medal_counts['Gold'] + medal_counts['Silver'] + medal_counts['Bronze']
medal_counts = medal_counts.sort_values(['Gold','Silver','Bronze'],ascending=False)

no_medals = medal_counts[medal_counts['Total']==0]
compete_data = pd.DataFrame(columns=['NOC', 'Sport', 'Compete'])
for country in no_medals.index:
    ang_medals = medal_counts[medal_counts.index == country]
    ang_athletes = atheletes[
        (atheletes['NOC'] == country) &
        (atheletes['Year'] >= 2020)
    ]

    times = 6
    for sport in ang_athletes['Sport'].unique():
        total_athletes = atheletes[(atheletes['Sport'] == sport) &
                                   (atheletes['Year'] >= 2024-times*4) &
                                   (atheletes['Medal'] != 'No medal')]
        team_counts = total_athletes['Team'].nunique()
        rate = times*3/team_counts
        compete_data = compete_data._append({'NOC': country, 'Sport': sport, 'Compete': rate}, ignore_index=True)
        print(f"{sport}: {team_counts}")
        print(f"Rate: {rate}")

compete_data['NOC'] = compete_data['NOC'].apply(noc_to_country)
compete_data['Total_Compete'] = compete_data.groupby('Sport')['Compete'].transform('sum')
compete_data['Medal_Prob'] = compete_data['Compete'] / compete_data['Total_Compete']
medal_probs = compete_data.groupby('NOC')['Medal_Prob'].sum().reset_index()
medal_probs.rename(columns={'Medal_Prob': 'Expected_Medals'}, inplace=True)
medal_probs = medal_probs.sort_values('Expected_Medals', ascending=False)
medal_probs.to_csv('./data/compete_prob.csv', index=False)

print(medal_probs)
