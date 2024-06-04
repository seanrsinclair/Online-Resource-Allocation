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

mean_size = 2
var_size = .1

# algo_list = ['hope_guardrail_14', 'hope_guardrail_12', 'hope_guardrail_13']
algo_list = ['static_x_lower', 'static_b_over_n', 'hope_guardrail_35', 'og_hope_guardrail_35']

# alpha_grid = [0.125, 0.15, 0.175, 0.2, 0.215, 0.25, 0.275]
# alpha_grid = [0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
alpha_grid = [0.3, 0.5, 0.7, 0.9]
# alpha_grid = [0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
# alpha_grid = []
# alpha_grid = [.07, .075, .08, .085, .09, .095]
# alpha_grid = [0.051, 0.06, 0.065, 0.31]

# alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

for alpha in alpha_grid:
    file_name = "geometric_perishing_2_"+str(alpha).replace('.','-')
    print(f'Running for: {file_name}')

    def perish_dist(b, n):
        val = np.minimum(n,np.random.geometric(p = (2*n)**((-1)*(1+alpha))))
        return val


    def demand_dist(n, mean_size, var_size=.1):
        size = np.maximum(0, np.random.normal(loc=mean_size, scale=np.sqrt(var_size), size=n).astype(int))
        return size

    df = pd.read_csv('./data/'+file_name+'.csv')


    # NEED TO FILTER THE ALGORITHMS PROPERLY IN THE DIFFERENT SETTINGS FOR THE TWO CASES
    # ALSO PRINT OUT THE INFORMATION FOR THE STOCKOUT PROBABILITIES PLOTS


    df['Log Value'] = np.log(df['Value'])
    df['Log NumGroups'] = np.log(df['NumGroups'])

    plt.style.use('PaperDoubleFig.mplstyle.txt')
    plt.rc('text.latex', preamble=r'\usepackage{amsfonts}')

    # Make some style choices for plotting 

    dashesStyles = [[3,1],
                [2,1,10,1],
                [4, 1, 1, 1, 1, 1],[1000,1],[8,2]]


    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    dash_styles = ["",
                (4, 1.5),
                (1, 1),
                (3, 1, 1.5, 1),
                (5, 1, 1, 1),
                (5, 1, 2, 1, 2, 1),
                (2, 2, 3, 1.5),
                (1, 2.5, 3, 1.2)]

    plt.rc('text', usetex=True)



    modified_algo_list = algo_list

    # modified_algo_list = algo_list
    print(modified_algo_list)

    # if INCLUDE_PUA:
    df = df[df['Algorithm'].isin(modified_algo_list)]


    df = df.replace({'static_x_lower':r'Static $\underline{X}$',
                        'static_b_over_n':r'Static $B / \overline{N}$',
                        'hope_guardrail_12':'Perish-Guardrail',
                        'og_hope_guardrail_12':'Vanilla-Guardrail',
                        'hope_guardrail_35':'Perish-Guardrail',
                        'og_hope_guardrail_35':'Vanilla-Guardrail'})

    fg, axs = plt.subplots(1,4, sharex='all', figsize=(20,6))
    sns.lineplot(x='NumGroups', y='Value', hue=
                'Algorithm', style = 'Algorithm', dashes = dashesStyles[0:len(algo_list)],
                        data=df[df.Norm == 'Counterfactual_Envy'], ax = axs[0], linewidth=4, palette = sns.color_palette("colorblind", len(algo_list)),
                ci = None)

    sns.lineplot(x='NumGroups', y='Value', hue='Algorithm', style = 'Algorithm', dashes = dashesStyles[0:len(algo_list)], 
                data=df[df.Norm == 'Waste'], ax = axs[1], linewidth=4, palette = sns.color_palette("colorblind", len(algo_list)),
                ci = None)

    sns.lineplot(x='NumGroups', y='Value', hue='Algorithm', style = 'Algorithm', dashes = dashesStyles[0:len(algo_list)], 
                data=df[df.Norm == 'Hindsight_Envy'], ax = axs[2], linewidth=4, palette = sns.color_palette("colorblind", len(algo_list)),
                ci = None)

    sns.lineplot(x='NumGroups', y='Value', hue='Algorithm', style = 'Algorithm', dashes = dashesStyles[0:len(algo_list)], 
                data=df[df.Norm == 'Perished_Un_Allocated'], ax = axs[3], linewidth=4, palette = sns.color_palette("colorblind", len(algo_list)),
                ci = None)

    axs[0].get_legend().remove()
    axs[0].set_ylabel(r'$\mathbb{E}[\Delta_{\textit{EF}}]$')
    axs[0].set_xlabel('Number of Rounds')

    axs[1].set_ylabel(r'$\mathbb{E}[\Delta_{\textit{efficiency}}]$')
    axs[1].set_xlabel('Number of Rounds')
    axs[1].get_legend().remove()

    axs[2].get_legend().remove()
    axs[2].set_ylabel(r'$\mathbb{E}[$Envy$]$')
    axs[2].set_xlabel('Number of Rounds')

    axs[3].get_legend().remove()
    axs[3].set_ylabel(r'$\mathbb{E}[$Spoilage$]$')
    axs[3].set_xlabel('Number of Rounds')


    fg.savefig('./figures/'+file_name+'.pdf', bbox_inches = 'tight',pad_inches = 0.01, dpi=900)

    # df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]

    max_n = df['NumGroups'].max()

# Filter the DataFrame to include only rows where 'Score' is the maximum
    df_group = df.loc[df['NumGroups'] == max_n]

    df_group = df_group[df_group.Norm == 'Stockout']
    df_group = df_group.groupby(['Algorithm'], as_index=False).agg(
                    {'Value':['mean','sem']})
    
    df_group.loc[:, ('Value', 'sem')] = df_group.loc[:, ('Value', 'sem')] * 1.96


    print(df_group)
    # print(print(df_group.to_latex(index=False,
    #               formatters={"name": str.upper},
    #               float_format="{:.1f}".format,
    #     )) )

    

    # legend = axs[2].legend(ncol = 5, loc= 'lower center', bbox_to_anchor=(-1, -.3, 0.5, 0.5))
    # print(len(legend.get_lines()))
    # [legend.get_lines()[i].set_linewidth(3) for i in range(len(legend.get_lines()))]

    # helper.export_legend(legend, filename="t_scaling_legend.pdf")

