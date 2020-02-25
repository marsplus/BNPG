import pickle
import seaborn
import argparse
import matplotlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('graph_type', type=str)
    args = parser.parse_args()
    graph_type = args.graph_type

    plt.style.use('seaborn-paper')
    seaborn.set_context('paper', font_scale=2.2)
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['font.weight'] = 'medium'

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    matplotlib.rc('font', **font)
    seaborn.set_style("whitegrid")
    plt.figure(figsize=(8, 4))
    data = []
    mapping = {'1_2': 1.2, '1_5': 1.5, '2': 2}
    for keyw in ['1_2', '1_5', '2']:
        with open('result/homogeneous_g_concave_beta_%s_%s.p' % (keyw, graph_type), 'rb') as fid:
            d = pickle.load(fid)
        tmp = pd.DataFrame(d)
        tmp['exponent'] = pd.Series([mapping[keyw]] * len(tmp), index=tmp.index)
        data.append(pd.DataFrame(tmp))
    data = pd.concat(data)
    error = data.groupby('exponent').sem().transpose() * 1.96
    data.groupby('exponent').mean().transpose().plot(kind='bar', figsize=(12,7), yerr=error)
    plt.legend(fontsize=14, title="$\\beta$")
    plt.xticks(rotation='horizontal', fontsize=40)
    plt.yticks(fontsize=40)
    plt.ylabel('investing ratio', fontweight='bold', fontsize=40)
    plt.xlabel('$\\alpha$',  fontweight='bold', fontsize=40)
    plt.tight_layout()
    plt.savefig('result/homogeneous_concave_%s.pdf' % graph_type)

