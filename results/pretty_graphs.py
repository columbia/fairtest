import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle


rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 22})

#
# Medical
#
PLOT_LABEL = 'a'
for context in [0, 26, 4, 1]:
    filename = 'medical_reg_plots/context_{}'.format(context)
    fig = pickle.load(file(filename + '.pkl'))

    plt.text(0.07, 0.9, '(' + PLOT_LABEL + ')', horizontalalignment='center',
             verticalalignment='center', transform=plt.gca().transAxes)
    PLOT_LABEL = chr(ord(PLOT_LABEL) + 1)

    plt.savefig(filename + '.png')


#
# Recommender (Movie Rating)
#

ticks = {0: [3.0, 3.5, 4.0, 4.5],
         1: [0.5, 1.0, 1.5, 2.0],
         2: [20, 30, 40, 50, 60]}

for key in ticks.keys():
    dir = 'movies_{}_plots'.format(key)

    for sens in ['Age', 'Gender']:
        filename = dir+'/' + sens + '/context_0'
        fig = pickle.load(file(filename + '.pkl'))
        plt.yticks(ticks[key])
        plt.savefig(filename + '.png')
