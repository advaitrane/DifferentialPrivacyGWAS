import numpy as np
import heapq
from tqdm import tqdm
from util import read_bed_for_snp, read_phenotype, get_num_SNPs

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

def allelic_test_nd(genotype_dist):
	x = (2*genotype_dist['r0'] + genotype_dist['r1'])
	y = (2*genotype_dist['s0'] + genotype_dist['s1'])

	value = 2*genotype_dist['N']*(x*genotype_dist['S'] - y*genotype_dist['R'])**2
	value /= (genotype_dist['R']*genotype_dist['S']*(x+y)*(2*genotype_dist['N']-x-y))
	return value

function_dict = {
	chi_squared_test: "Chi squared test",
	allelic_test: "Allelic test",
	allelic_test_nd: "Allelic test"
}

def top_SNPs_standard(
	data_dir, 
	genotype_file_name, 
	phenotype_file_name,
	statistic_function,
	M,
	num_SNPs = None
):
	print(f"Running the standard algorithm to calculate top {M} SNPs by the {function_dict[statistic_function]}")
	if num_SNPs is None:
		num_SNPs = get_num_SNPs(data_dir, genotype_file_name)

	M_SNPs_heap = []

	phenotype_list = read_phenotype(data_dir, phenotype_file_name)
	print(f"Calculating statistics for {num_SNPs} SNPs")
	for idx_SNP in tqdm(range(num_SNPs)):
		genotype_list = read_bed_for_snp(data_dir, genotype_file_name, idx_SNP)
		genotype_dist = get_genotype_dist(genotype_list, phenotype_list)

		value = statistic_function(genotype_dist)
		if len(M_SNPs_heap) < M:
			heapq.heappush(M_SNPs_heap, (value, idx_SNP))
		else:
			if value > M_SNPs_heap[0][0]:
				heapq.heapreplace(M_SNPs_heap, (value, idx_SNP))

	return M_SNPs_heap