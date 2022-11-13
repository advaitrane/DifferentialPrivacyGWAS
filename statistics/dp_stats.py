import numpy as np
import heapq
from util import read_bed_for_snp, read_phenotype
from statistics.standard_stats import get_genotype_dist

def chi_squared_test_sensitivity(genotype_dist):
	
	"""
	Paper says, "If we have no knowledge of either the cases or the controls, 
	we get the sensitivity result presented in Corollary 3.5. On the other 
	hand, when the controls are known, we can use Theorem 3.4 to reduce the 
	sensitivity assigned to each set of SNPs grouped by the maximum number of 
	controls among the three genotypes."
	"""


	# This will result in the same sensitivity for all SNPS
	# if (
	# 	genotype_dist['r0'] >= genotype_dist['r2'] and 
	# 	genotype_dist['s0'] >= genotype_dist['s2']
	# 	):
	#	# Corollary 3.5
	# 	m = max(genotype_dist['R'], genotype_dist['S'])
	# else:
	#	# Theorem 3.4
	#	m = max(genotype_dist['s0'], genotype_dist['s1'], genotype_dist['s2'])

	# This will result in different sensivity for different SNPs
	m = max(genotype_dist['s0'], genotype_dist['s1'], genotype_dist['s2'])

	sensitivity = (
			genotype_dist['N']**2/(genotype_dist['R']*genotype_dist['S'])
			)*(
			1 - 1/(m + 1)
			)

	return sensitivity

def allelic_test_sensitivity(genotype_dist):
	N = genotype_dist['N']
	R = genotype_dist['R']
	S = genotype_dist['S']

	v1 = (8* N**2 * S)/(R*(2*S+3)*(2*S+1))
	v2 = (4* N**2 * ((2*R**2-1)*(2*S-1) - 1))/(R*S*(2*R+1)*(2*R-1)*(2*S+1))
	v3 = (8* N**2 * R)/(S*(2*R+3)*(2*R+1))
	v4 = (4* N**2 * ((2*S**2-1)*(2*R-1) - 1))/(R*S*(2*S+1)*(2*S-1)*(2*R+1))

	sensitivity = max(v1, v2, v3, v4)
	return sensitivity

def top_SNPs_laplace(
	DATA_DIR, 
	genotype_file_name, 
	phenotype_file_name,
	statistic_function,
	sensitivity_function,
	M,
	epsilon,
	num_SNPs = None
):
	
	# TODO
	if num_SNPs is None:
		# get num snps from file
		num_SNPs = M + M

	M_SNPs_heap = []

	phenotype_list = read_phenotype(DATA_DIR, phenotype_file_name)
	for idx_SNP in range(num_SNPs):
		genotype_list = read_bed_for_snp(DATA_DIR, genotype_file_name, idx_SNP)
		genotype_dist = get_genotype_dist(genotype_list, phenotype_list)

		value = statistic_function(genotype_dist)
		sensitivity = sensitivity_function(genotype_dist)

		scale = 4*M*sensitivity/epsilon
		perturbed_statistic_value = value + np.random.laplace(0, scale)

		if len(M_SNPs_heap) < M:
			heapq.heappush(M_SNPs_heap, (perturbed_statistic_value, idx_SNP))
		else:
			if perturbed_statistic_value > M_SNPs_heap[0][0]:
				heapq.heapreplace(M_SNPs_heap, ((perturbed_statistic_value, idx_SNP)))

	M_SNPs = []
	for _ in range(M):
		perturbed_value, idx = heapq.heappop(M_SNPs_heap)
		genotype_list = read_bed_for_snp(DATA_DIR, genotype_file_name, idx)
		genotype_dist = get_genotype_dist(genotype_list, phenotype_list)

		value = statistic_function(genotype_dist)
		sensitivity = sensitivity_function(genotype_dist)
		scale = 2*M*sensitivity/epsilon
		laplace_noise = np.random.laplace(0, scale)
		print(laplace_noise)
		perturbed_value = value + laplace_noise
		M_SNPs.append((perturbed_value, idx))

	return M_SNPs

def top_SNPs_exponential(
	DATA_DIR, 
	genotype_file_name, 
	phenotype_file_name,
	statistic_function,
	sensitivity_function,
	M,
	epsilon,
	num_SNPs = None
):
	if num_SNPs is None:
		# get num snps from file
		num_SNPs = M + M

	w_SNPs = np.zeros(num_SNPs)
	w_sum = 0
	phenotype_list = read_phenotype(DATA_DIR, phenotype_file_name)
	for idx_SNP in range(num_SNPs):
		genotype_list = read_bed_for_snp(DATA_DIR, genotype_file_name, idx_SNP)
		genotype_dist = get_genotype_dist(genotype_list, phenotype_list)

		value = statistic_function(genotype_dist)
		sensitivity = sensitivity_function(genotype_dist)

		w = np.exp((epsilon*value)/(4*M*sensitivity))
		w_sum += w
		w_SNPs[idx_SNP] = w
	w_SNPs /= w_sum

	M_SNPs = []
	for _ in range(M):
		idx_SNP = np.random.choice(np.arange(num_SNPs), p = w_SNPs)
		genotype_list = read_bed_for_snp(DATA_DIR, genotype_file_name, idx_SNP)
		genotype_dist = get_genotype_dist(genotype_list, phenotype_list)

		value = statistic_function(genotype_dist)
		sensitivity = sensitivity_function(genotype_dist)
		scale = 2*M*sensitivity/epsilon
		laplace_noise = np.random.laplace(0, scale)
		perturbed_value = value + laplace_noise
		M_SNPs.append((perturbed_value, idx_SNP))

		w_SNPs *= w_sum
		w_sum -= w_SNPs[idx_SNP]
		w_SNPs[idx_SNP] = 0
		w_SNPs /= w_sum

	return M_SNPs





