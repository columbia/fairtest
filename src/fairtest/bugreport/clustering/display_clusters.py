from StringIO import StringIO
import prettytable

from fairtest.bugreport.statistics import fairness_measures as fm

class NodeFilter:
    LEAVES_ONLY = 1
    LEAVES_AND_ROOT = 2
    ALL = 3


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
def bug_report(clusters, measure='MI', sort_by='sig', node_filter=NodeFilter.LEAVES_ONLY, conf_level=0.95):
    assert measure in ['MI', 'CORR']
    assert sort_by in ['sig', 'effect']

    # take only the leaves
    if node_filter == NodeFilter.LEAVES_ONLY:
        clusters = filter(lambda c: c.isleaf, clusters)
    elif node_filter == NodeFilter.LEAVES_AND_ROOT:
        clusters = filter(lambda c: c.isleaf or c.isroot, clusters)

    if measure == 'MI':
        # compute p-values and mutual information CIs for all clusters
        p_vals = map(lambda c: fm.G_test(c.stats)[1], clusters)
        effects = map(lambda c: fm.mutual_info(c.stats, ci_level=conf_level), clusters)
    else:
        # TODO
        p_vals = [0]*len(clusters)
        effects = map(lambda c: fm.correlation(c.stats, ci_level=conf_level), clusters)

    zipped = zip(clusters, p_vals, effects)
        
    # sort by significance
    if sort_by == 'sig':
        zipped.sort(key=lambda tup: tup[1])
    # sort by effect-size
    elif sort_by == 'effect':
        zipped.sort(key=lambda tup: max(0, tup[2][0]-tup[2][1]), reverse=True)
    
    # print clusters in order of relevance    
    for (cluster, p_val, (effect, effect_delta)) in zipped:
        #ctype = "LEAF" if cluster.isleaf else "ROOT" if cluster.isroot else "INTERNAL"
        #print '{} node {} of size {}'.format(ctype, cluster.num, cluster.size)
        print 'Population of size {}'.format(cluster.size)
        print 'Context = {}'.format(cluster.path)
        print

        if measure == 'MI':
            # pretty-print the contingency table
            output = StringIO()
            rich_ct(cluster.stats).to_csv(output)
            output.seek(0)
            pt = prettytable.from_csv(output)
            print pt
            print
        
            # print p-value and confidence interval of MI
            mi_low = max(0, effect-effect_delta)
            mi_high = effect+effect_delta
            print 'p-value = {:.2e} ; MI = [{:.4f}, {:.4f}]'.format(p_val, mi_low, mi_high)
        else:
            # print p-value and confidence interval of correlation
            corr_low = max(0, effect-effect_delta)
            corr_high = min(effect+effect_delta, 1)
            print 'p-value = {:.2e} ; Corr = [{:.4f}, {:.4f}]'.format(p_val, corr_low, corr_high)
        print '-'*80
        print

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
