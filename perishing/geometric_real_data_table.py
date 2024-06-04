import warnings;
warnings.filterwarnings('ignore');

from time import sleep
from tqdm.auto import tqdm

import helper

import sys
import importlib
import numpy as np
import nbformat
# import plotly.express
# import plotly.express as px
import pandas as pd
import cvxpy as cp
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import seaborn as sns
import helper
import algorithms


INCLUDE_PUA = True


# algo_list = ['hope_guardrail_14', 'hope_guardrail_12', 'hope_guardrail_13']
# algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_12', 'og_hope_guardrail_12']
algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_35', 'og_hope_guardrail_35']


# alpha_grid = [.07, .075, .08, .085, .09, .095]
# alpha_grid = [0.051, 0.06, 0.065, 0.31]

# alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9]


file_name = "geometric_perishing_2_real_data_"
print(f'Running for: {file_name}')

df = pd.read_csv('./data/'+file_name+'.csv')


# NEED TO FILTER THE ALGORITHMS PROPERLY IN THE DIFFERENT SETTINGS FOR THE TWO CASES
# ALSO PRINT OUT THE INFORMATION FOR THE STOCKOUT PROBABILITIES PLOTS


df['Log Value'] = np.log(df['Value'])
df['Log NumGroups'] = np.log(df['NumGroups'])




modified_algo_list = algo_list

# modified_algo_list = algo_list
print(modified_algo_list)

# if INCLUDE_PUA:
df = df[df['Algorithm'].isin(modified_algo_list)]

algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_12', 'og_hope_guardrail_12']


df = df.replace({'static_x_lower':r'Static $\underline{X}$',
                    'static_b_over_n':r'Static $B / \overline{N}$',
                    'hope_guardrail_12':'Perish-Guardrail',
                    'og_hope_guardrail_12':'Vanilla-Guardrail'})


# df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]
df_group = df
df_group = df_group.groupby(['Algorithm', 'Norm'], as_index=False).agg(
                {'Value':['mean', 'sem']})
# df_group['sem'] = 1.96*df_group['sem']

# print(df_group.columns)
# print(df_group.info())

df_group.loc[:, ('Value', 'sem')] = df_group.loc[:, ('Value', 'sem')] * 1.96


# df_group.loc[df['Value'].index.get_level_values(1) == 'sem', ('Value', 'sem')] *= 1.96



tmp = pd.pivot_table(df_group, index='Algorithm', columns = 'Norm')

print(tmp)


print(print(tmp.to_latex(index=True,
              formatters={"name": str.upper},
              float_format="{:.1f}".format,
    )) )