import numpy as np
import math
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

def get_neighbour_distance(genotype_dist, w, Y):
	# def point_on_C(genotype_dist, x, y, w):
	# 	lhs = 2*genotype_dist['N']*(
	# 		x*genotype_dist['S'] - y*genotype_dist['R']
	# 		)**2
	# 	rhs = genotype_dist['R']*genotype_dist['S']*w*(x + y)*(
	# 		2*genotype_dist['N'] - x - y
	# 		)
	# 	return lhs == rhs

	N = genotype_dist['N']
	R = genotype_dist['R']
	S = genotype_dist['S']
	r0 = genotype_dist['r0']
	r1 = genotype_dist['r1']
	r2 = genotype_dist['r2']
	s0 = genotype_dist['s0']
	s1 = genotype_dist['s1']
	s2 = genotype_dist['s2']

	def get_P():
		P = set()
		m_vals = [1, 2, 0.5]

		for m in m_vals:
			alpha = 2*N*R*S*m - R*S*w*m - 2*N*S*S - R*S*w
			beta = N*R*S*w*(1+m)
			gamma = 2*N*R*R*m + R*S*w*m + R*S*w - 2*N*R*S
			alpha /= gamma
			beta /= gamma

			A = 2*N*(S-alpha*R)**2 + R*S*w*(1+alpha)**2
			B = 4*N*beta*R*(alpha*R-S) - 2*R*S*w*(N - beta)*(1 + alpha)
			C = 2*N*beta*beta*R*R - R*S*w*beta*(2*N - beta)

			disc = B*B - 4*A*C
			if disc >= 0:
				x = (-B + np.sqrt(disc))/(2*A)
				y = alpha*x + beta
				if not (np.isnan(x) or math.isinf(x)):
					if x>=0 and x<=2*R and y>=0 and y<=2*S:
						P.add((x, y))

				x = (-B - np.sqrt(disc))/(2*A)
				y = alpha*x + beta
				if not (np.isnan(x) or math.isinf(x)):
					if x>=0 and x<=2*R and y>=0 and y<=2*S:
						P.add((x, y))
		return P

	def get_y_for_x(x):
		A = 2*N*R*R + R*S*w
		B = 2*R*S*w*x - 2*N*R*S*w - 4*N*R*S*x
		C = 2*N*S*S*x*x + R*S*w*x*x - 2*N*R*S*w*x

		d = B*B - 4*A*C

		if d >= 0:
			y1 = (-B + np.sqrt(d)) / (2*A)
			y2 = (-B - np.sqrt(d)) / (2*A)
			return y1, y2
		return None, None

	def get_x_for_y(y):
		A = 2*N*S*S + R*S*w
		B = 2*R*S*w*y - 2*N*R*S*w - 4*N*R*S*y
		C = 2*N*S*S*y*y + R*S*w*y*y - 2*N*R*S*w*y

		d = B*B - 4*A*C

		if d >= 0:
			x1 = (-B + np.sqrt(d)) / (2*A)
			x2 = (-B - np.sqrt(d)) / (2*A)
			return x1, x2
		return None, None

	def get_Q():
		Q = set()
		x_vals = [
			2*(genotype_dist['r0'] + genotype_dist['r2']) + genotype_dist['r1'],
			2*genotype_dist['r0'] + genotype_dist['r1'],
			genotype_dist['r1'],
			0,
			2*R
		]
		y_vals = [
			2*(genotype_dist['s0'] + genotype_dist['s2']) + genotype_dist['s1'],
			2*genotype_dist['s0'] + genotype_dist['s1'],
			genotype_dist['s1'],
			0,
			2*S
		]

		for x in x_vals:
			y1, y2 = get_y_for_x(x)
			if (y1 is not None and y1 >= 0 and y1 <= 2*S and not (np.isnan(y1) or math.isinf(y1))):
				Q.add((x, y1))
			if (y2 is not None and y2 >= 0 and y2 <= 2*S and not (np.isnan(y2) or math.isinf(y2))):
				Q.add((x, y2))

		for y in y_vals:
			x1, x2 = get_x_for_y(y)
			if (x1 is not None and x1 >= 0 and x1 <= 2*R and not (np.isnan(x1) or math.isinf(x1))):
				Q.add((x1, y))
			if (x2 is not None and x2 >= 0 and x2 <= 2*R and not (np.isnan(x2) or math.isinf(x2))):
				Q.add((x2, y))

		return Q

	def g1_x(x):
		if (x <= 2*R - r1 and 
			x >= 2*r0 + r1
		):
			return (x-2*r0-r1)/2
		elif (x >= r1 and 
			x <= 2*r0+r1
		):
			return (2*r0+r1-x)/2
		elif (x <= 2*R and x >= 2*R - r1):
			return r2+x-2*(r0+r2)-r1
		else:
			return r0+r1-x

	def g2_y(y):
		if (y <= 2*S - s1 and 
			y >= 2*s0 + s1
		):
			return (y-2*s0-s1)/2
		elif (y >= s1 and 
			y <= 2*s0+s1
		):
			return (2*s0+s1-y)/2
		elif (y <= 2*S and y >= 2*S - s1):
			return s2+y-2*(s0+s2)-s1
		else:
			return s0+s1-y

	def g_xy(x, y):
		return g1_x(x) + g2_y(y)

	def beta1_x(x):
		if (r1 == 0 and 
			(x-2*r0-r1)%2 == 1
		):
			return np.ceil(g1_x(x))+1
		else:
			return np.ceil(g1_x(x))

	def beta2_y(y):
		if (s1 == 0 and 
			(y-2*s0-s1)%2 == 1
		):
			return np.ceil(g2_y(y))+1
		else:
			return np.ceil(g2_y(y))

	def get_limits_for_Ui(i):
		if r1>0 and s1>0:
			if i == 0 or i == 1:
				return (2*r0+r1, 2*(r0+r2)+r1)
			if i == 2 or i == 3:
				return (r1, 2*r0+r1)
			if i == 4 or i == 5:
				return (2*(r0+r2)+r1, 2*R)
			if i == 6 or i == 7:
				return (0, r1)

	def get_limits_for_Uprimej(j, alpha, beta):
		if r1>0 and s1>0:
			if j == 0 or j == 1:
				l1 = (2*s0+s1-beta)/alpha
				l2 = (2*(s0+s2)+s1-beta)/alpha
			if j == 2 or j == 3:
				l1 = (s1-beta)/alpha
				l2 = (2*s0+s1-beta)/alpha
			if j == 4 or j == 5:
				l1 = (2*(s0+s2)+s1-beta)/alpha
				l2 = (2*S-beta)/alpha
			if j == 6 or j == 7:
				l1 = (-beta)/alpha
				l2 = (s1-beta)/alpha

			l_min = min(l1, l2)
			l_max = max(l1, l2)
			return l_min, l_max

	def get_alpha_beta(i, j, delta):
		vi_list = [1/2, 1/2 ,-1/2, -1/2, 1, 1, -1, -1]
		di_list = [-r0-r1/2, -r0-r1/2+1/2, r0+r1/2, r0+r1/2+1/2, r2-2*(r0+r2)-r1, r2-2*(r0+r2)-r1, r0+r1, r0+r1]

		vj_list = [1/2, 1/2 ,-1/2, -1/2, 1, 1, -1, -1]
		dj_list = [-s0-s1/2, -s0-s1/2+1/2, s0+s1/2, s0+s1/2+1/2, s2-2*(s0+s2)-s1, s2-2*(s0+s2)-s1, s0+s1, s0+s1]

		vi = vi_list[i]
		di = di_list[i]
		vj = vj_list[j]
		dj = dj_list[j]

		alpha = -vi/vj
		beta = (delta - di - dj)/vj

		return alpha, beta

	def get_limits_i_j(i, j, delta):
		alpha, beta = get_alpha_beta(i, j, delta)

		l1, l2 = get_limits_for_Ui(i)
		l3, l4 = get_limits_for_Uprimej(j, alpha, beta)
		
		if l2<l3 or l4<l1:
			return None, None

		if l3>l1:
			l1 = l3
		if l4<l2:
			l2 = l4

		return l1, l2

	def get_limits_Y(alpha, beta):
		A = 2*N*(S-alpha*R)**2 + R*S*w*(1+alpha)**2
		B = 4*N*beta*R*(alpha*R-S) - 2*R*S*w*(N - beta)*(1 + alpha)
		C = 2*N*beta*beta*R*R - R*S*w*beta*(2*N - beta)

		d = B*B - 4*A*C
		if d >= 0:
			l1 = (-B - np.sqrt(d))/(2*A)
			l2 = (-B + np.sqrt(d))/(2*A)
			return l1, l2
		return None, None

	def get_r_s(i, j, delta):
		if r1>0 and s1>0:
			l1, l2 = get_limits_i_j(i, j, delta)
			alpha, beta = get_alpha_beta(i, j, delta)
			l3, l4 = get_limits_Y(alpha, beta)

			if l1 is None or l2 is None or l3 is None or l4 is None:
				return None, None
			if l2<l3 or l4<l1:
				return None, None
			r = max(l1, l3)
			s = min(l2, l4)
			return r, s

	def check_delta(delta):
		if r1>0 and s1>0:
			ci_vals = [-2*r0-r1, -2*r0-r1+1, -2*r0-r1, -2*r0-r1+1, 0, 1, 0, 1]
			cj_vals = [-2*s0-s1, -2*s0-s1+1, -2*s0-s1, -2*s0-s1+1, 0, 1, 0, 1]

			for i in range(8):
				for j in range(8):
					alpha, beta = get_alpha_beta(i, j, delta)
					r, s = get_r_s(i, j, delta)

					if r is None:
						continue
					
					r = np.ceil(r)

					for k in range(5):
						t = r+k
						x = t
						y = alpha*t + beta

						ci = x+ci_vals[i]
						cj = y+cj_vals[j]

						if ci%2==0 and cj%2==0:
							return True
			return False
		else:
			print(f"r1 and s1 don't match condition - ({r1}, {s1})")

	P = get_P()
	Q = get_Q()

	PQ = P.union(Q)
	g_hat = None
	for x, y in PQ:
		g_val = np.ceil(g_xy(x, y))
		if g_hat is None or g_hat > g_val:
			g_hat = g_val

	if Y < w:
		return g_hat

	for i in range(6):
		delta = g_hat+i
		if check_delta(delta):
			return delta

	return None










