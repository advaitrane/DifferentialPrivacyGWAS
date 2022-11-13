import numpy as np

def get_genotype_dist(genotype_list, phenotype_list):
	genotype_dist = {v:0 for v in [
	'r0', 'r1', 'r2', 'R', 's0', 's1', 's2', 'S', 'n0', 'n1', 'n2', 'N'
	]}

	num_subjects = len(genotype_list)
	for subject_idx in range(num_subjects):
		g = genotype_list[subject_idx]
		if g < 0:
			continue
		p = phenotype_list[subject_idx]

		p2d_map = {1:'s', 2:'r'}
		genotype_dist[p2d_map[p]+(str)(g)] += 1
		genotype_dist[p2d_map[p].upper()] += 1
		genotype_dist['n'+(str)(g)] += 1
		genotype_dist['N'] += 1

	return genotype_dist

def chi_squared_test(genotype_dist):
	value = 0
	if genotype_dist['n0'] > 0:
		value += (
			genotype_dist['r0']*genotype_dist['N'] - genotype_dist['n0']*genotype_dist['R']
			)**2 / (genotype_dist['n0']*genotype_dist['R']*genotype_dist['S'])
	if genotype_dist['n1'] > 0:
		value += (
			genotype_dist['r1']*genotype_dist['N'] - genotype_dist['n1']*genotype_dist['R']
			)**2 / (genotype_dist['n1']*genotype_dist['R']*genotype_dist['S'])
	if genotype_dist['n2'] > 0:
		value += (
			genotype_dist['r2']*genotype_dist['N'] - genotype_dist['n2']*genotype_dist['R']
			)**2 / (genotype_dist['n2']*genotype_dist['R']*genotype_dist['S'])
	return value

def allelic_test(genotype_dist):
	x = (genotype_dist['s1']+2*genotype_dist['s2'])
	y = (genotype_dist['n1']+2*genotype_dist['n2'])

	value = (2 * genotype_dist['N']**3)/(genotype_dist['R']*genotype_dist['S'])
	value *= (x - (genotype_dist['S']/genotype_dist['N'])*y)**2
	value /= (2*genotype_dist['N']*y - y**2)
	return value