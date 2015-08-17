from StringIO import StringIO
from math import log
import prettytable
from statsmodels.sandbox.stats.multicomp import multipletests


from fairtest.bugreport.statistics import fairness_measures as fm
from fairtest.bugreport.statistics.fairness_measures import NMI, CORR

class NodeFilter:
    LEAVES_ONLY = 1
    ALL = 2


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
def bug_report(clusters, columns=None, measure=NMI(ci_level=0.95), sort_by='sig', node_filter=NodeFilter.LEAVES_ONLY, fdr=None):
    assert isinstance(measure, fm.Measure)
    assert sort_by in ['sig', 'effect']

    # compute effect sizes and p-values
    stats = map(lambda c: measure.compute(c.stats), clusters)
    
    # corrected p-values
    if fdr:
        pvals = map(lambda (low, high, p): max(p, 1e-180), stats)
        _, pvals_corr, _, _ = multipletests(pvals, alpha=fdr, method='holm')
        stats = [measure.ci_from_p(low, high, pval_corr) for ((low, high, _), pval_corr) in zip(stats, pvals_corr)]
    
    zipped = zip(clusters, stats)
    
    # take only the leaves (plus the root)
    if node_filter == NodeFilter.LEAVES_ONLY:
        zipped = filter(lambda (c, stat): c.isleaf or c.isroot, zipped)
    
    # print global stats
    (root, (root_effect_low, root_effect_high, root_pval)) = filter(lambda (c, _): c.isroot, zipped)[0]
    print 'Global Population of size {}'.format(root.size)
    print
    
    # print a contingency table or correlation analysis
    if not isinstance(measure, fm.CORR):
        print_cluster_ct(root, columns, root_pval, root_effect_low, root_effect_high, measure.__class__.__name__)
    else:
        print_cluster_corr(root, root_pval, root_effect_low, root_effect_high, measure.__class__.__name__)
    print '='*80
    print
    
    zipped = filter(lambda (c, _): not c.isroot, zipped)
    
        
    # sort by significance
    if sort_by == 'sig':
        zipped.sort(key=lambda (c, (low,high,p)): p)
    # sort by effect-size
    elif sort_by == 'effect':
       zipped.sort(key=lambda (c, stats): measure.normalize_effect(stats), reverse=True)
        
        
    # print clusters in order of relevance    
    for (cluster, (effect_low, effect_high, p_val)) in zipped:
        print 'Sub-Population of size {}'.format(cluster.size)
        print 'Context = {}'.format(cluster.path)
        print

        if not isinstance(measure, fm.CORR):
            print_cluster_ct(cluster, columns, p_val, effect_low, effect_high, measure.__class__.__name__)
        else:
            print_cluster_corr(cluster, p_val, effect_low, effect_high, measure.__class__.__name__)
        print '-'*80
        print

def print_cluster_ct(cluster, columns, p_val, effect_low, effect_high, effect_name):
    # pretty-print the contingency table
    output = StringIO()
    
    if columns:
        ct = cluster.stats[columns]
    else:
        ct = cluster.stats
    
    rich_ct(ct).to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    print pt
    print 
    
    # print p-value and confidence interval of MI
    print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.format(p_val, effect_name, effect_low, effect_high)

def print_cluster_corr(cluster, p_val, effect_low, effect_high, effect_name):
    # print p-value and confidence interval of correlation
    print 'p-value = {:.2e} ; {} = [{:.4f}, {:.4f}]'.format(p_val, effect_name, effect_low, effect_high)
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
