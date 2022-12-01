import os
import util
from util import read_bed_for_snp, read_phenotype, get_dataset, get_SNP_name, get_causal_SNPS, get_metrics
from statistics.standard_stats import get_genotype_dist, chi_squared_test, allelic_test, allelic_test_nd, top_SNPs_standard
from statistics.dp_stats import top_SNPs_laplace, top_SNPs_exponential, chi_squared_test_sensitivity, allelic_test_sensitivity
from statistics.dp_stats import get_neighbour_distance, top_SNPs_neighbour_distance
from statistics.dp_models import DPLogisticRegression

import numpy as np
import torch
from tqdm import tqdm
import time


DATA_DIR = "../data/hapmap_JPT_CHB_r23a_filtered"
# file_name = "hapmap_JPT_CHB_r23a_filtered"
# phenotype_file_name = "phenotype_35_2"

def DP_vs_standard(
	data_dir, 
	genotype_file_name, 
	phenotype_file_name,
	statistic_name,
	output_dir
):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	M_values = [3, 5, 10, 20]
	epsilon_values = [1, 5, 10, 50, 100, 500, 1000]
	total_num_SNPs = None
	stat_func_dict = {'allelic_test': allelic_test, 'chi_squared_test': chi_squared_test}
	statistic_function = stat_func_dict[statistic_name]
	sensitivity_func_dict = {'allelic_test': allelic_test_sensitivity, 'chi_squared_test': chi_squared_test_sensitivity}
	sensitivity_function = sensitivity_func_dict[statistic_name]

	standard_file_path = os.path.join(output_dir, "standard.log")
	laplace_file_path = os.path.join(output_dir, "laplace.log")
	exponential_file_path = os.path.join(output_dir, "exponential.log")
	nd_file_path = os.path.join(output_dir, "nd.log")

	for M in M_values:
		if M > 0:
			start_time = time.time()
			M_SNPS_standard = top_SNPs_standard(
				data_dir, 
				genotype_file_name, 
				phenotype_file_name,
				statistic_function,
				M,
				total_num_SNPs
				)
			standard_time = time.time() - start_time
			# print("Top SNPs given by the standard algorithm")
			M_SNPS_standard = [
				(get_SNP_name(data_dir, genotype_file_name, b), round(a, 3)) for (a, b) in M_SNPS_standard
				]
			with open(standard_file_path, 'a+') as f:
				f.write(f"{M}\t{M_SNPS_standard}\t{standard_time}\n")

		for epsilon in epsilon_values:
			# if M == 3 and epsilon < 1000:
			# 	continue
			print(f"M = {M}, epsilon = {epsilon}")

			start_time = time.time()
			M_SNPS_laplace = top_SNPs_laplace(
				data_dir, 
				genotype_file_name, 
				phenotype_file_name,
				statistic_function,
				sensitivity_function,
				M,
				epsilon,
				total_num_SNPs
				)
			laplace_time = time.time() - start_time
			# print("Top SNPs given by the laplace algorithm")
			M_SNPS_laplace = [
				(get_SNP_name(data_dir, genotype_file_name, b), round(a, 3)) for (a, b) in M_SNPS_laplace
				]
			with open(laplace_file_path, 'a+') as f:
				f.write(f"{M}\t{epsilon}\t{M_SNPS_laplace}\t{laplace_time}\n")

			start_time = time.time()
			M_SNPS_exp = top_SNPs_exponential(
				data_dir, 
				genotype_file_name, 
				phenotype_file_name,
				statistic_function,
				sensitivity_function,
				M,
				epsilon,
				total_num_SNPs
				)
			exp_time = time.time() - start_time
			# print("Top SNPs given by the exponential algorithm")
			M_SNPS_exp = [
				(get_SNP_name(data_dir, genotype_file_name, b), round(a, 3)) for (a, b) in M_SNPS_exp
				]
			with open(exponential_file_path, 'a+') as f:
				f.write(f"{M}\t{epsilon}\t{M_SNPS_exp}\t{exp_time}\n")


			start_time = time.time()
			M_SNPS_nd = top_SNPs_neighbour_distance(
				data_dir, 
				genotype_file_name, 
				phenotype_file_name,
				statistic_function,
				sensitivity_function,
				get_neighbour_distance,
				M,
				epsilon,
				total_num_SNPs
			)
			nd_time = time.time() - start_time
			# print("Top SNPs given by the neighbour distance algorithm")
			M_SNPS_nd = [
				(get_SNP_name(data_dir, genotype_file_name, b), round(a, 3)) for (a, b) in M_SNPS_nd
				]
			with open(nd_file_path, 'a+') as f:
				f.write(f"{M}\t{epsilon}\t{M_SNPS_nd}\t{nd_time}\n")

def eval_DP_logistic_regression(
	data_dir, 
	genotype_file_name, 
	phenotype_file_name,
	output_dir,
	n_epochs,
	snp_idxs
):
	x, y = get_dataset(data_dir, genotype_file_name, phenotype_file_name, 
		snp_idxs
		)

	train_test_split_point = (int)(0.8*len(x))
	x_train = torch.from_numpy(x[:train_test_split_point].astype(np.float32))
	x_val = torch.from_numpy(x[train_test_split_point:].astype(np.float32))
	y_train = torch.from_numpy(y[:train_test_split_point].astype(np.float32))
	y_val = torch.from_numpy(y[train_test_split_point:].astype(np.float32))

	
	epsilon_values = [1, 5, 10, 50, 100, 500, 1000]
	reg_lambda_values = [1e-2, 5e-3, 1e-3, 5e-4]
	reg_alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	output_file_path = os.path.join(output_dir, "dp_model.log")

	for epsilon in epsilon_values:
		for reg_lambda in reg_lambda_values:
			for reg_alpha in reg_alpha_values:
				print(f"epsilon = {epsilon}, reg_lambda = {reg_lambda}, reg_alpha = {reg_alpha}")
				dplr = DPLogisticRegression(5, reg_lambda, reg_alpha, epsilon)
				dplr.set_psi(x_train)

				opt = torch.optim.SGD(dplr.parameters(), 1e-2)

				for epoch in range(n_epochs):
					dplr.train()
					opt.zero_grad()
					
					logits = dplr(x_train)
					train_loss = dplr.logistic_loss(y_train, logits)
					reg_loss = dplr.regularization_loss(len(y_train))
					loss = train_loss+reg_loss

					loss.backward()
					opt.step()

					dplr.eval()
					logits_val = dplr(x_val)
					val_loss = dplr.logistic_loss(y_val, logits_val) + dplr.regularization_loss(len(y_val))
					y_pred_val = dplr.pred(x_val)
					val_metrics = get_metrics(y_val, y_pred_val)

					with open(output_file_path, 'a+') as f:
						f.write(f"{epsilon}\t{reg_lambda}\t{reg_alpha}\t{epoch}\t{loss}\t{val_metrics[0]}\t{val_metrics[1]}\t{val_metrics[2]}\t{val_loss}\n")

def main():

	# genotype_file_name = "hapmap_JPT_CHB_r23a_filtered"
	# phenotype_file_name = "phenotype_35_2"

	# DP_vs_standard(
	# 	DATA_DIR,
	# 	genotype_file_name,
	# 	phenotype_file_name,
	# 	'chi_squared_test',
	# 	'output/35_2_chi_squared_test_xxx'
	# 	)


	# n_epochs = 50
	# snp_idxs = [2112391, 415960, 1821253, 1821244, 1821273]
	# eval_DP_logistic_regression(
	# 	DATA_DIR, 
	# 	genotype_file_name, 
	# 	phenotype_file_name,
	# 	'output/35_2_50_dp_model_xxx',
	# 	n_epochs,
	# 	snp_idxs
	# 	)

	"""
	snp_idx = 214
	genotype_list_0 = read_bed_for_snp(DATA_DIR, file_name, snp_idx)

	phenotype_list = read_phenotype(DATA_DIR, phenotype_file_name)

	genotype_dist = get_genotype_dist(genotype_list_0, phenotype_list)
	print(f"Genotype distribution for SNP {snp_idx} - {genotype_dist}")
	
	# chi_squared_value = chi_squared_test(genotype_dist)
	# print(f"Chi squared value for SNP {snp_idx} - {chi_squared_value}")
	allelic_test_value = allelic_test(genotype_dist)
	print(f"Allelic test value for SNP {snp_idx} - {allelic_test_value}")
	allelic_test_nd_value = allelic_test_nd(genotype_dist)
	print(f"Allelic test (ND) value for SNP {snp_idx} - {allelic_test_nd_value}")

	causal_SNPs = get_causal_SNPS(DATA_DIR, phenotype_file_name)
	"""

	"""
	M_SNPS_standard = top_SNPs_standard(
		DATA_DIR, 
		file_name, 
		phenotype_file_name,
		allelic_test,
		10
		)
	print("Top SNPs given by the standard algorithm")
	M_SNPS_standard = [
		(a, get_SNP_name(DATA_DIR, file_name, b)) for (a, b) in M_SNPS_standard
		]
	print(M_SNPS_standard)

	count = 0
	for a, b in M_SNPS_standard:
		if b in causal_SNPs:
			count += 1

	print("Standard causal count = ", count)

	M_SNPS_laplace = top_SNPs_laplace(
		DATA_DIR, 
		file_name, 
		phenotype_file_name,
		allelic_test,
		allelic_test_sensitivity,
		10,
		1000
		)
	print("Top SNPs given by the laplace algorithm")
	M_SNPS_laplace = [
		(a, get_SNP_name(DATA_DIR, file_name, b)) for (a, b) in M_SNPS_laplace
		]
	print(M_SNPS_laplace)

	count = 0
	for a, b in M_SNPS_laplace:
		if b in causal_SNPs:
			count += 1

	print("Laplace causal count = ", count)


	M_SNPS_exp = top_SNPs_exponential(
		DATA_DIR, 
		file_name, 
		phenotype_file_name,
		allelic_test,
		allelic_test_sensitivity,
		10,
		1000		)
	print("Top SNPs given by the exponential algorithm")
	M_SNPS_exp = [
		(a, get_SNP_name(DATA_DIR, file_name, b)) for (a, b) in M_SNPS_exp
		]
	print(M_SNPS_exp)

	count = 0
	for a, b in M_SNPS_exp:
		if b in causal_SNPs:
			count += 1

	print("Exponential causal count = ", count)

	# d = get_neighbour_distance(genotype_dist, 10, allelic_test_value)
	# print(f'Neighbour distance = {d}')

	M_SNPS_nd = top_SNPs_neighbour_distance(
		DATA_DIR, 
		file_name, 
		phenotype_file_name,
		allelic_test,
		allelic_test_sensitivity,
		get_neighbour_distance,
		10,
		100
	)
	print("Top SNPs given by the neighbour distance algorithm")
	M_SNPS_nd = [
		(a, get_SNP_name(DATA_DIR, file_name, b)) for (a, b) in M_SNPS_nd
		]
	print(M_SNPS_nd)

	count = 0
	for a, b in M_SNPS_nd:
		if b in causal_SNPs:
			count += 1

	print("ND causal count = ", count)
	"""
	"""
	x, y = get_dataset(DATA_DIR, file_name, phenotype_file_name, 
		[160107, 557579, 588795, 708065, 747570]
		)

	x_train = torch.from_numpy(x[:70].astype(np.float32))
	x_val = torch.from_numpy(x[70:].astype(np.float32))
	y_train = torch.from_numpy(y[:70].astype(np.float32))
	y_val = torch.from_numpy(y[70:].astype(np.float32))

	# print(x.shape, y.shape)
	# print(x[:5, :])
	# print(y)

	dplr = DPLogisticRegression(5, 1e-2, 0.5, 10)
	dplr.set_psi(x_train)

	opt = torch.optim.SGD(dplr.parameters(), 1e-2)

	n_epochs = 10
	for epoch in range(n_epochs):
		opt.zero_grad()
		
		logits = dplr(x_train)
		train_loss = dplr.logistic_loss(y_train, logits)
		reg_loss = dplr.regularization_loss(len(y_train))
		loss = train_loss+reg_loss

		loss.backward()
		opt.step()

		print(loss)
		val_logits = dplr(x_val)
		print(val_logits)
		print(y_val)
	"""

if __name__ == '__main__':
	main()