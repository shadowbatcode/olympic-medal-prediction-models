import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
describe = data[['Gold','Silver','Bronze','Total','Rank']].describe()


describe.to_csv('describe.csv')