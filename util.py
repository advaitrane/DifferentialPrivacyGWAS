import numpy as np
import os

def num_lines_in_file(file_path):
	def _count_generator(reader):
	    b = reader(1024 * 1024)
	    while b:
	        yield b
	        b = reader(1024 * 1024)

	with open(file_path, 'rb') as fp:
	    c_generator = _count_generator(fp.raw.read)
	    # count each \n
	    count = sum(buffer.count(b'\n') for buffer in c_generator)

	return count

def read_bed_for_snp(data_dir, file_name, snp_idx):
	""" Read distribution of alleles for given snp idx from bed file. Return
	a list of -1/0/1/2 to indicate number of minor alleles for each subject.
	"""
	bed_file_path = os.path.join(data_dir, file_name + ".bed")
	bim_file_path = os.path.join(data_dir, file_name + ".bim")
	fam_file_path = os.path.join(data_dir, file_name + ".fam")

	num_subjects = num_lines_in_file(fam_file_path)
	num_bytes_per_snp = (int)(np.ceil(num_subjects/4))
	file_idx = 3 + num_bytes_per_snp*snp_idx
	genotype_list = []
	num_subjects_read = 0
	with open(bed_file_path, "rb") as f:
		f.seek(file_idx)
		for _ in range(num_bytes_per_snp):
			data = f.read(1)[0]
			for _ in range(4):
				code = data & 0b11
				data = data >> 2
				if code == 0b00:
					genotype = 2
				elif code == 0b01:
					genotype = -1
				elif code == 0b10:
					genotype = 1
				elif code == 0b11:
					genotype = 0
				else:
					print("Error in reading file - unknown bit combination")
					genotype = -2

				genotype_list.append(genotype)
				num_subjects_read += 1

				if num_subjects_read == num_subjects:
					break
	return genotype_list

def read_phenotype(data_dir, file_name):
	pheno_file_path = os.path.join(data_dir, file_name + ".pheno")
	phenotype_list = []
	with open(pheno_file_path, 'r') as f:
		l = f.readline()
		for line in f:
			phenotype = (int)(line.strip().split('\t')[-1])
			phenotype_list.append(phenotype)

	return phenotype_list

def get_SNP_name(data_dir, file_name, SNP_idx):
	bim_file_path = os.path.join(data_dir, file_name + ".bim")

	with open(bim_file_path, 'r') as f:
		for line_idx, line in enumerate(f):
			if line_idx == SNP_idx:
				SNP_name = line.split('\t')[1]
	return SNP_name

def get_causal_SNPS(data_dir, phenotype_file_name):
	causals_file_path = os.path.join(data_dir, phenotype_file_name + ".1.causals")

	causal_SNPs = []
	with open(causals_file_path, 'r') as f:
		for idx, line in enumerate(f):
			if idx == 0:
				continue
			causal_SNPs.append(line.split("\t")[0])

	return causal_SNPs

def get_num_SNPs(data_dir, genotype_file_name):
	return num_lines_in_file(os.path.join(data_dir, genotype_file_name + ".bim"))

def get_dataset(data_dir, genotype_file_name, phenotype_file_name, snp_idxs):
	x = []

	for snp_idx in snp_idxs:
		x.append(read_bed_for_snp(data_dir, genotype_file_name, snp_idx))

	x = np.array(x).T

	y = np.array(read_phenotype(data_dir, phenotype_file_name))
	y = 2*y - 3

	drop_rows = (x != -1).all(1)
	x = x[drop_rows]
	y = y[drop_rows]

	return x, y

def get_metrics(y_true, y_pred):
	acc = round((y_true == y_pred).sum().item()/len(y_pred), 3)

	pos_pred = y_pred[y_pred == 1]
	pos_true = y_true[y_pred == 1]
	tp = (pos_pred == pos_true).sum().item()
	fp = len(pos_pred) - tp
	neg_pred = y_pred[y_pred == -1]
	neg_true = y_true[y_pred == -1]
	tn = (neg_pred == neg_true).sum().item()
	fn = len(neg_pred) - tn

	if tp == 0:
		precision = 0
		recall = 0
	else:
		precision = round(tp/(tp+fp), 3)
		recall = round(tp/(tp+fn), 3)

	return acc, precision, recall

