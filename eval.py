import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from util import get_causal_SNPS

DATA_DIR = "../data/hapmap_JPT_CHB_r23a_filtered"

def eval_causal_SNPs(causal_SNPs, top_SNPs):
	num_causal_SNPs = len(causal_SNPs)

	num_SNPs_detected = 0
	for SNP in causal_SNPs:
		if SNP in top_SNPs:
			num_SNPs_detected += 1

	return num_SNPs_detected/num_causal_SNPs

def eval_IOU(standard_SNPs, top_SNPs):
	standard_set = set(standard_SNPs)
	top_set = set(top_SNPs)

	n = len(standard_set.intersection(top_set))
	d = len(standard_set.union(top_set))
	return n/d

def plot_field(
	standard_logs, 
	top_SNP_logs, 
	field, 
	M_vals, 
	epsilon_vals,
	alg_vals,
	fig_file_name,
	fig_title
):
	fig = plt.figure(figsize=(10, 5))
	diff = 0.2
	leg = []
	for M in M_vals:
		for epsilon in epsilon_vals:
			leg.append(f"M={M}, e={epsilon}")
			values = []
			if field in standard_logs[M]:
				values.append(standard_logs[M][field])
			for alg in alg_vals:
				values.append(top_SNP_logs[alg][M][epsilon][field])
			x_vals = np.arange(len(values)).astype(float)+1
			x_vals -= diff
			diff -= 0.4/(len(M_vals)*len(epsilon_vals))

			sns.lineplot(x=x_vals, y=values)
	plt.legend(leg, bbox_to_anchor=(1.2, 1), loc='upper right')
	if len(x_vals) == len(alg_vals)+1:
		x_labels = ['standard'] + alg_vals
	else:
		x_labels = alg_vals
	plt.xticks(np.arange(len(values))+1, x_labels)
	plt.xlabel('Algorithm')
	plt.ylabel(field)
	plt.title(fig_title)
	plt.savefig(fig_file_name, bbox_inches='tight')
	plt.show()

def eval_top_SNPs(
	data_dir,
	log_dir,
	phenotype_file_name,
	fig_title_prefix
):
	causals_file_path = os.path.join(data_dir, phenotype_file_name + ".1.causals")
	causal_SNPs = get_causal_SNPS(data_dir, phenotype_file_name)

	standard_log_file = "standard.log"
	top_SNPs_log_files = ["laplace.log", "exponential.log", "nd.log"]

	standard_logs = {}
	with open(os.path.join(log_dir, standard_log_file), 'r') as f:
		for line in f:
			sp = line.split('\t')
			M = (int)(sp[0])
			top_SNPs = [v[2:-1] for v in sp[1].strip('][').split(', ') if v[2:4] == 'rs']
			time = (float)(sp[2])
			
			standard_logs[M] = {}
			standard_logs[M]['SNPs'] = top_SNPs
			standard_logs[M]['time'] = time
			standard_logs[M]['causal_SNP_prop'] = eval_causal_SNPs(causal_SNPs, top_SNPs)

	top_SNP_algs = ['laplace', 'exponential', 'nd'] # 
	top_SNP_logs = {}
	for idx, log_file in enumerate(top_SNPs_log_files):
		log_file_path = os.path.join(log_dir, log_file)
		if idx >= len(top_SNP_algs):
			break
		alg = top_SNP_algs[idx]
		top_SNP_logs[alg] = {M:{} for M in standard_logs.keys()}
		with open(log_file_path, 'r') as f:
			for line in f:
				sp = line.split('\t')
				M = (int)(sp[0])
				epsilon = (int)(sp[1])
				top_SNPs = [v[2:-1] for v in sp[2].strip('][').split(', ') if v[2:4] == 'rs']
				time = (float)(sp[3])

				top_SNP_logs[alg][M][epsilon] = {}
				# top_SNP_logs[alg][M][epsilon]['SNPs'] = top_SNPs
				top_SNP_logs[alg][M][epsilon]['time'] = time
				top_SNP_logs[alg][M][epsilon]['causal_SNP_prop'] = eval_causal_SNPs(
					causal_SNPs, top_SNPs
					)
				top_SNP_logs[alg][M][epsilon]['standard_SNP_IOU'] = eval_IOU(
					standard_logs[M]['SNPs'], top_SNPs
					)

	Ms = [3, 5, 10, 20] # [3, 5, 10, 20]
	Es = [500] # [1, 5, 10, 50, 100, 500, 1000]
	fields = ['causal_SNP_prop', 'standard_SNP_IOU'] # ['causal_SNP_prop', 'standard_SNP_IOU', 'time']
	for field in fields:
		fig_file_name = f"{field}_M{Ms[0]}-{Ms[-1]}_E{Es[0]}-{Es[-1]}"
		fig_dir = os.path.join(log_dir, "results")
		if not os.path.isdir(fig_dir):
			os.makedirs(fig_dir)
		fig_file_name = os.path.join(fig_dir, fig_file_name)
		fig_title = fig_title_prefix + " " + field.replace('_', ' ')
		plot_field(
			standard_logs, 
			top_SNP_logs, 
			field, 
			Ms, 
			Es,
			top_SNP_algs,
			fig_file_name,
			fig_title
		)
	

def eval_dp_model(log_dir, epsilon_vals, reg_lamda_vals, alpha_vals, num_epochs):
	log_file_path = os.path.join(log_dir, "dp_model.log")

	metrics = ['loss', 'val_loss', 'accuracy', 'precision', 'recall']
	model_logs = {e:{
		rl:{
			a:{
				m:[] for m in metrics
				} for a in alpha_vals
			} for rl in reg_lamda_vals
		} for e in epsilon_vals}
	plot_on = {e:{
		rl:{
			a: False for a in alpha_vals
			} for rl in reg_lamda_vals
		} for e in epsilon_vals}
	with open(log_file_path) as f:
		for line in f:
			sp = line.split('\t')
			e = (int)(sp[0])
			l = (float)(sp[1])
			a = (float)(sp[2])
			ep = (int)(sp[3])

			loss = float(sp[4])
			accuracy = float(sp[5])
			precision = float(sp[6])
			recall = float(sp[7])
			val_loss = float(sp[8])

			if ep == num_epochs-1:
				if accuracy > 0.7 and precision > 0.5 and recall > 0.5:
					plot_on[e][l][a] = True

			if ep >= num_epochs:
				continue

			if e in epsilon_vals and l in reg_lamda_vals and a in alpha_vals:
				model_logs[e][l][a]['loss'].append(loss)
				model_logs[e][l][a]['val_loss'].append(val_loss)
				model_logs[e][l][a]['accuracy'].append(accuracy)
				model_logs[e][l][a]['precision'].append(precision)
				model_logs[e][l][a]['recall'].append(recall)


	# for m in metrics:
	# 	leg = []
	# 	for e in epsilon_vals:
	# 		for l in reg_lamda_vals:
	# 			for a in alpha_vals:
	# 				if not plot_on[e][l][a]:
	# 					continue
	# 				y_vals = model_logs[e][l][a][m]
	# 				# print(y_vals)
	# 				sns.lineplot(x=np.arange(len(y_vals)), y=y_vals)
	# 				leg.append(f"epsilon={e},lamda={l},alpha={a}")

	# 	plt.legend(leg) # , bbox_to_anchor=(1.5, 1), loc='upper right'
	# 	plt.xlabel('epochs')
	# 	plt.ylabel(m)
	# 	plt.title(f"{m} for best performing models")
	# 	plt.show()

	# for e in epsilon_vals:
	# 	for l in reg_lamda_vals:
	# 		for m in ['precision', 'recall']:
	# 			y_vals = []
	# 			for a in alpha_vals:
	# 				y_vals.append(max(model_logs[e][l][a][m]))
	# 			sns.lineplot(x=alpha_vals, y=y_vals)
	# 			plt.xlabel("alpha")
	# 			plt.ylabel(m)
	# 			plt.title(f"epsilon={e}, lamda={l}")
	# 			plt.show()

		for m in ['accuracy', 'precision', 'recall']:
			displace = {e:-0.125 for e in epsilon_vals}
			df_plot = pd.DataFrame(columns=['e', 'm', 'l', 'a'])
			df_idx = 0
			for e in epsilon_vals:
				for l in reg_lamda_vals:
					for a in alpha_vals: 
						if plot_on[e][l][a]:
							df_plot.loc[df_idx] = [
								np.log(e)+displace[e], max(model_logs[e][l][a][m]), l, a
								]
							df_idx += 1
							displace[e] += 0.125

			sns.scatterplot(data=df_plot, x='e', y='m', hue='a', style='l')

			xt = [np.log(e) for e in epsilon_vals]
			plt.xticks(xt, epsilon_vals)
			plt.xlabel("epsilon")
			plt.ylabel(m)
			plt.title(f"Best values for {m}")
			plt.legend(bbox_to_anchor=(1.2, 1), loc='upper right')
			fig_file_name = f"best_{m}"
			fig_file_name = os.path.join(log_dir, fig_file_name)
			plt.savefig(fig_file_name, bbox_inches='tight')
			plt.show()

def main():
	# log_dir = "output/35_2_allelic_test"
	# phenotype_file_name = "phenotype_35_2"
	# ld_sp = log_dir.split('/')[1].split('_')
	# fig_title_prefix = f"Allelic test (k=0.{ld_sp[0]}, n={ld_sp[1]})"
	# eval_top_SNPs(
	# 	DATA_DIR, 
	# 	log_dir, 
	# 	phenotype_file_name,
	# 	fig_title_prefix
	# )

	log_dir = "output/35_2_50_dp_model"
	epsilon_values = [1, 5, 10, 50, 100, 500, 1000]
	reg_lambda_values = [1e-2, 5e-3, 1e-3, 5e-4]
	reg_alpha_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	num_epochs = 50
	eval_dp_model(
		log_dir, epsilon_values, reg_lambda_values, reg_alpha_values, num_epochs
		)





if __name__ == '__main__':
	main()