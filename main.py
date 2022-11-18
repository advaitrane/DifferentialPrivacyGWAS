import os
import util
from util import read_bed_for_snp, read_phenotype
from statistics.standard_stats import get_genotype_dist, chi_squared_test, allelic_test
from statistics.dp_stats import top_SNPs_laplace, top_SNPs_exponential, chi_squared_test_sensitivity, allelic_test_sensitivity
from statistics.dp_stats import get_neighbour_distance

DATA_DIR = "../data/hapmap_JPT_CHB_r23a_filtered"
file_name = "hapmap_JPT_CHB_r23a_filtered"
phenotype_file_name = "simu.pheno"

def main():
	snp_idx = 219
	genotype_list_0 = read_bed_for_snp(DATA_DIR, file_name, snp_idx)

	phenotype_list = read_phenotype(DATA_DIR, phenotype_file_name)

	genotype_dist = get_genotype_dist(genotype_list_0, phenotype_list)
	print(f"Genotype distribution for SNP {snp_idx} - {genotype_dist}")
	
	# chi_squared_value = chi_squared_test(genotype_dist)
	# print(f"Chi squared value for SNP {snp_idx} - {chi_squared_value}")
	allelic_test_value = allelic_test(genotype_dist)
	print(f"Allelic test value for SNP {snp_idx} - {allelic_test_value}")

	# M_SNPS_laplace = top_SNPs_laplace(
	# 	DATA_DIR, 
	# 	file_name, 
	# 	phenotype_file_name,
	# 	chi_squared_test,
	# 	chi_squared_test_sensitivity,
	# 	10,
	# 	1000,
	# 	1000
	# 	)
	# print("Top SNPs given by the laplace algorithm")
	# print(M_SNPS_laplace)

	# M_SNPS_exp = top_SNPs_exponential(
	# 	DATA_DIR, 
	# 	file_name, 
	# 	phenotype_file_name,
	# 	chi_squared_test,
	# 	chi_squared_test_sensitivity,
	# 	10,
	# 	1000,
	# 	1000
	# 	)
	# print("Top SNPs given by the exponential algorithm")
	# print(M_SNPS_exp)

	d = get_neighbour_distance(genotype_dist, 10, allelic_test_value)
	print(f'Neighbour distance = {d}')

if __name__ == '__main__':
	main()