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


INCLUDE_PUA = True

mean_size = 2
var_size = .1

# algo_list = ['static_x_lower']
# order_list = ['mean', 'cv', 'random', 'reverse']

algo_list = ['hope_guardrail_35']
# algo_list = ['static_x_lower']
order_list = ['mean', 'cv', 'lcb', 'ucb']

# order_list = ['mean', 'reverse', 'random']
problem_list = ['A', 'B']




# algo_list = ['static_x_lower', 'static_b_over_n']
# order_list = ['mean', 'cv', 'random', 'reverse']
# order_list = ['mean', 'reverse']


for algo in algo_list:
    # print(algo)
    for setup in problem_list: # useless for loop at the moment, in case we want to run for different alpha variables

        file_name = "uniform_perishing_"+str(setup).replace('.','-')
        print(f'Running for: {file_name} and algorithm: {algo}')
        df = pd.read_csv('./data/'+file_name+'.csv')


        # NEED TO FILTER THE ALGORITHMS PROPERLY IN THE DIFFERENT SETTINGS FOR THE TWO CASES
        # ALSO PRINT OUT THE INFORMATION FOR THE STOCKOUT PROBABILITIES PLOTS

        df = df[df['Algorithm'].str.startswith(algo)]
        print(df['Algorithm'].unique())

        df['Log Value'] = np.log(df['Value'])
        df['Log NumGroups'] = np.log(df['NumGroups'])





        df = df.replace({'mean':r'Increasing $\mathbb{E}[T_b]$',
                        'flipped':r'Decreasing $\mathbb{E}[T_b]$',
                        'random':r'Random',
                        'lcb':r'Lower Confidence Bound',
                        'ucb':r'Upper Confidence Bound',
                        'cv': r"Decreasing CV"})
        

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



        # modified_algo_list = algo_list

        # modified_algo_list = algo_list
        # print(modified_algo_list)

        # if INCLUDE_PUA:
        # df = df[df['Algorithm'].isin(modified_algo_list)]


        fg, axs = plt.subplots(1,4, sharex='all', figsize=(20,6))

        sns.lineplot(x='NumGroups', y='Value', hue=
                    'Order', style = 'Order', dashes = dashesStyles[0:len(order_list)],
                            data=df[df.Norm == 'Counterfactual_Envy'], ax = axs[0], linewidth=4, palette = sns.color_palette("colorblind", len(order_list)),
                    ci = None)

        sns.lineplot(x='NumGroups', y='Value', hue='Order', style = 'Order', dashes = dashesStyles[0:len(order_list)], 
                    data=df[df.Norm == 'Waste'], ax = axs[1], linewidth=4, palette = sns.color_palette("colorblind", len(order_list)),
                    ci = None)

        sns.lineplot(x='NumGroups', y='Value', hue='Order', style = 'Order', dashes = dashesStyles[0:len(order_list)], 
                    data=df[df.Norm == 'Hindsight_Envy'], ax = axs[2], linewidth=4, palette = sns.color_palette("colorblind", len(order_list)),
                    ci = None)

        sns.lineplot(x='NumGroups', y='Value', hue='Order', style = 'Order', dashes = dashesStyles[0:len(order_list)], 
                    data=df[df.Norm == 'Perished_Un_Allocated'], ax = axs[3], linewidth=4, palette = sns.color_palette("colorblind", len(order_list)),
                    ci = None)

        # axs[0].get_legend().remove()
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


        fg.savefig('./figures/'+file_name+"_"+algo+'.pdf', bbox_inches = 'tight',pad_inches = 0.01, dpi=900)

        # df_group = df[df.NumGroups >= (1/4) * max(df.NumGroups)]

    #     max_n = df['NumGroups'].max()

    # # Filter the DataFrame to include only rows where 'Score' is the maximum
    #     df_group = df.loc[df['NumGroups'] == max_n]

    #     df_group = df_group[df_group.Norm == 'Stockout']
    #     df_group = df_group.groupby(['Algorithm'], as_index=False).agg(
    #                     {'Value':['mean','std']})

    #     print(df_group)
        

    legend = axs[2].legend(ncol = 4, loc= 'lower center', bbox_to_anchor=(-1, -.3, 0.5, 0.5))
    print(len(legend.get_lines()))
    [legend.get_lines()[i].set_linewidth(3) for i in range(len(legend.get_lines()))]

    helper.export_legend(legend, filename="t_scaling_legend_normal.pdf")


