load('/Users/craggles/OneDrive - University Of Cambridge/Programming/Python/PhD/flOPT/helicity_H_recovered_tx.mat')

f = boxplot(pc_sum_trans)
export_pretty_fig('pc_sum_trans',f)