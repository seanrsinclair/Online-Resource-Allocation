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
# algo_list = ['hope_guardrail_35']


# alpha_grid = [.07, .075, .08, .085, .09, .095]
# alpha_grid = [0.051, 0.06, 0.065, 0.31]

# alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9]


# problem_list = ['A','B','C']
problem_list = ['C']

for setup in problem_list:
    file_name = "uniform_perishing_table_"+str(setup).replace('.','-')

    print(f'Running for: {file_name}')

    df = pd.read_csv('./data/'+file_name+'.csv')
    df = df.drop('Algorithm', axis=1)
    print(df['NumGroups'].unique())
    # df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]
    grouped_df = df.groupby(['Order', 'Norm']).agg({'Value': ['mean', 'sem']}).reset_index()
    grouped_df[('Value', 'sem')] *= 1.96

    # grouped_df.loc[:, ('Value', 'sem')] = grouped_df.loc[:, ('Value', 'sem')] * 1.96


    # df_group.loc[df['Value'].index.get_level_values(1) == 'sem', ('Value', 'sem')] *= 1.96


    # print(grouped_df)
    tmp = pd.pivot_table(grouped_df, index='Order', columns = 'Norm')

    # print(tmp)


    print(print(tmp.to_latex(index=True,
                formatters={"name": str.upper},
                float_format="{:.4f}".format,
        )) )