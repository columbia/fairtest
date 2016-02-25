import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle


rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 22})

"""
#
# Medical
#
PLOT_LABEL = 'a'
for context in [0, 26, 4, 1]:
    filename = 'medical_reg_plots/context_{}'.format(context)
    fig = pickle.load(file(filename + '.pkl'))

    #plt.text(0.07, 0.9, '(' + PLOT_LABEL + ')', horizontalalignment='center',
    #         verticalalignment='center', transform=plt.gca().transAxes)
    #PLOT_LABEL = chr(ord(PLOT_LABEL) + 1)

    plt.savefig(filename + '.png')
"""

#
# Recommender (Movie Rating)
#

ticks = {0: [3.6, 3.8, 4.0, 4.2, 4.4, 4.6],
         1: [0, 0.5, 1.0, 1.5, 2.0],
         2: [10, 20, 30, 40, 50, 60, 70]}

for key in ticks.keys():
    dir = 'movies_{}_plots'.format(key)

    for sens in ['Age', 'Gender']:
        filename = dir+'/' + sens + '/context_0'
        fig = pickle.load(file(filename + '.pkl'))
        plt.yticks(ticks[key])
        plt.ylim(ticks[key][0], ticks[key][-1])
        plt.savefig(filename + '.png')
