from fairtest.bugreport.statistics import fairness_measures as fm
import numpy as np
import pandas as pd
from StringIO import StringIO
import prettytable 

#
# Print all the clusters sorted by relevance
#
# @args clusters    list of all clusters
# @args sort_by     way to sort the clusters ('effect' => sort by lower bound
#                   on the mutual information score; 'sig' => sort by significance
#                   level)
# @args leaves_only consider tree leaves only
# @args conf_level  level for confidence intervals
#
def bug_report(clusters, sort_by='sig', leaves_only=True, conf_level=0.95):
    
    # take only the leaves
    if leaves_only:
        clusters = filter(lambda c: c.isleaf, clusters)
    
    # compute p-values and mutual information CIs for all clusters
    p_vals = map(lambda c: fm.G_test(c.ct)[1], clusters)
    mi_s = map(lambda c: fm.mutual_info(c.ct, norm=False, ci=True, level=conf_level), clusters)
    
    zipped = zip(clusters, p_vals, mi_s)
        
    # sort by significance
    if sort_by == 'sig':
        zipped.sort(key=lambda tup: tup[1])
    # sort by effect-size
    elif sort_by == 'effect':
        zipped.sort(key=lambda tup: max(0, tup[2][0]-tup[2][1]), reverse=True)
    
    # print clusters in order of relevance    
    for (cluster, p_val, (mi, mi_delta)) in zipped:
        ctype = "LEAF" if (cluster.isleaf) else "ROOT" if (cluster.isroot) else "INTERNAL"
        print '{} node {} of size {}'.format(ctype, cluster.num, cluster.size)
        print 'Context = {}'.format(cluster.path)
        print
        
        # pretty-print the contingency table
        output = StringIO()
        rich_ct(cluster.ct).to_csv(output)
        output.seek(0)
        pt = prettytable.from_csv(output)
        
        print pt
        print
        
        # print p-value and confidence interval of MI
        mi_low = max(0, mi-mi_delta)
        mi_high = mi+mi_delta
        print 'p-value = {:.2e} ; MI = [{:.4f}, {:.4f}]'.format(p_val, mi_low, mi_high)
        print '-'*80

#
# Build a rich contingency table with proportions and marginals
#     
# @args ct  the contingency table
#   
def rich_ct(ct):

    # for each output, add its percentage over each sensitive group
    temp = ct.copy().astype(object)
    for col in ct.columns:
        tot_col = sum(ct[col])
        for row in ct.index:
            val = ct.loc[row][col]
            percent = (100.0*val)/tot_col
            temp.loc[row][col] = '{} ({:.1f}%)'.format(val, percent)
    
    total = ct.sum().sum()
    
    # add marginals
    sum_col = ct.sum(axis=1)
    temp.insert(len(temp.columns), 'Total', map(lambda val: '{} ({:.1f}%)'.format(val, (100.0*val)/total) , sum_col))
    sum_row = ct.sum(axis=0)
    sum_row['Total'] = total
    temp.loc['Total'] = map(lambda val: '{} ({:.1f}%)'.format(val, (100.0*val)/total) , sum_row)
    temp.loc['Total']['Total'] = '{} (100.0%)'.format(total)
    
    return temp