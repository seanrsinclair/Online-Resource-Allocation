import warnings;
warnings.filterwarnings('ignore');

from time import sleep
from tqdm.auto import tqdm

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


algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail', 'og_hope_guardrail']

plt.style.use('PaperDoubleFig.mplstyle.txt')
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')


# alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.25]
alpha_grid = [0.1]
# alpha_grid = [0.2, 0.25]
# alpha_grid = [0.1]
l_val = 0.35

for alpha in alpha_grid:

    file_name = "geometric_perishing_tradeoff_"+str(alpha).replace('.','-')
    print(f'Running for: {file_name}')


    df = pd.read_csv('./data/'+file_name+'.csv')

    df_static_lower = df[df['Algorithm'] == 'static_x_lower']
    df_static_bn = df[df['Algorithm'] == 'static_b_over_n']
    df_hope = df[df['Algorithm'] == 'hope_guardrail']
    df_og = df[df['Algorithm'] == 'og_hope_guardrail']
    df_cusp = df[df['Algorithm'].isin(['hope_guardrail', 'og_hope_guardrail'])] # filtering by the algorithms


    df_static_lower = df_static_lower.drop(['NumGroups', 'Algorithm'], axis=1)
    tmp_static_lower = df_static_lower.groupby(['Norm']).mean()
    tmp_static_lower = pd.pivot_table(tmp_static_lower, values='Value', columns='Norm')
    tmp_static_lower['Envy Param'] = tmp_static_lower.index


    df_static_bn = df_static_bn.drop(['NumGroups', 'Algorithm'], axis=1)
    tmp_static_bn = df_static_bn.groupby(['Norm']).mean()
    tmp_static_bn = pd.pivot_table(tmp_static_bn, values='Value', columns='Norm')
    tmp_static_bn['Envy Param'] = tmp_static_bn.index


    df_hope = df_hope.drop(['NumGroups', 'Algorithm'], axis=1)
    tmp_hope = df_hope.groupby(['Envy Param', 'Norm']).mean()
    tmp_hope = pd.pivot_table(tmp_hope, index='Envy Param', values='Value', columns='Norm')
    tmp_hope['Envy Param'] = tmp_hope.index

    df_og = df_og.drop(['NumGroups', 'Algorithm'], axis=1)
    tmp_og = df_og.groupby(['Envy Param', 'Norm']).mean()
    tmp_og = pd.pivot_table(tmp_og, index='Envy Param', values='Value', columns='Norm')
    tmp_og['Envy Param'] = tmp_og.index

    df_cusp = df_cusp.drop('NumGroups', axis=1) # getting rid of unnecessary column
    df_cusp = df_cusp[df_cusp['Envy Param'] == l_val] # get out that particular L_T value we want to star.
    df_cusp = df_cusp.drop('Envy Param', axis=1)
    tmp_cusp = df_cusp.groupby(['Norm', 'Algorithm']).mean()
    tmp_cusp = pd.pivot_table(tmp_cusp, index='Algorithm', values='Value', columns=['Norm'])

    plt.rc('text', usetex=True)
    fg, axs = plt.subplots(1,1, sharex='all', figsize=(6,4))

    # sns.scatterplot(x='Counterfactual_Envy', y='Waste', data=tmp_og, hue='Envy Param', palette = 'summer', ax = axs, legend=False)
    # sns.scatterplot(x='Counterfactual_Envy', y='Waste', data=tmp_hope, hue='Envy Param', palette = 'cool', ax = axs, legend=False)
    sns.lineplot(x='Counterfactual_Envy', y='Waste', data=tmp_og, estimator=max, color='green', linestyle='dashed', ci=None, alpha='Envy Param', label=r'Vanilla-Guardrail')
    sns.lineplot(x='Counterfactual_Envy', y='Waste', data=tmp_hope, estimator=max, color = 'blue', linestyle = 'dotted', ci=None, label='Perish-Guardrail')
    sns.scatterplot(x='Counterfactual_Envy', y='Waste', data=tmp_static_bn, color='g', ax = axs, s=100, label=r'Static $B/\overline{N}$')
    sns.scatterplot(x='Counterfactual_Envy', y='Waste', data=tmp_static_lower, color = 'b', ax = axs, s=100, label=r'Static $\underline{X}$')
    sns.scatterplot(x='Counterfactual_Envy', y='Waste', data=tmp_cusp, marker='*', color = 'k', ax = axs, s=400, label=r'$L_T = T^{-0.35}$')


    plt.ylim(0, 105)
    plt.xlim(0, 1.8)

    plt.ylabel(r'$\mathbb{E}[\Delta_{\textit{efficiency}}]$')
    plt.xlabel(r'$\mathbb{E}[\Delta_{\textit{EF}}]$')
    # plt.show()
    axs.get_legend().remove()

    fg.savefig('./figures/'+file_name+'.pdf', bbox_inches = 'tight',pad_inches = 0.01, dpi=900)
    
    # plt.show()

    legend = axs.legend(ncol = 5, loc= 'lower center', bbox_to_anchor=(-1, -1.3, 0.5, 0.5))
    print(len(legend.get_lines()))
    [legend.get_lines()[i].set_linewidth(3) for i in range(len(legend.get_lines()))]

    helper.export_legend(legend, filename="tradeoff_legend.pdf")

        # df_static_lower = df_static_lower.groupby(['Metric'], as_index=False).agg(
        #                 {'Value':['mean','std']})
        
        # tmp = tmp.groupby(['Algorithm', 'Norm']).mean()
        # vals = pd.pivot_table(tmp, index='Algorithm', values='Value', columns='Norm')
        # vals['Algorithm'] = vals.index


        # # df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]
        # df_group =
        # df_group = df_group[df_group.Norm == 'Stockout']
        # df_group = df_group.groupby(['Algorithm'], as_index=False).agg(
        #                 {'Value':['mean','std']})

        # print(df_group)
        # # print(print(df_group.to_latex(index=False,
        # #               formatters={"name": str.upper},
        # #               float_format="{:.1f}".format,
        # #     )) )

