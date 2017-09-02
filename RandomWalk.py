__author__ = 'lbernardi'
import random as rand
import numpy as np
import math
import matplotlib.pyplot as plt

def as_vector(state):
    v = np.zeros(5)
    v[state-1]=1.
    return v

def walk():
    states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    initial_state = states.index('D')
    current_state = initial_state
    path = []
    path.append(as_vector(current_state))
    while True:
        if rand.random()<0.5:
            current_state = current_state + 1
        else:
            current_state = current_state - 1

        if states[current_state] == 'G':
            return path, 1.
        if states[current_state] == 'A':
            return path, 0.
        path.append(as_vector(current_state))




def td_cumulative(training_set, l, alpha):
	w = np.random.rand(5)
	w = np.ones(5)*0.5
	for sequence, outcome in training_set:
		e = sequence[0]
		delta_ws = []	
		for t, x_t in enumerate(sequence):
			if t == len(sequence)-1:
				P_tp1 = outcome
			else:
				x_tp1 = sequence[t+1]
				P_tp1 = np.dot(x_tp1, w)

			P_t = np.dot(x_t, w)
			cost = 1.
			delta_w_t = alpha*(cost+P_tp1-P_t)*e
	    		delta_ws.append(delta_w_t)
			
			if t < len(sequence)-1:
				e = sequence[t+1] + l*e
			
		delta_w = np.sum(delta_ws, axis=0)
		w += delta_w
	return w


def td_intra_sequence(training_set, l, alpha):
	w = np.random.rand(5)
	w = np.ones(5)*0.5
	for sequence, outcome in training_set:
		e = sequence[0]
		for t, x_t in enumerate(sequence):
			if t == len(sequence)-1:
				P_tp1 = outcome
			else:
				x_tp1 = sequence[t+1]
				P_tp1 = np.dot(x_tp1, w)

			P_t = np.dot(x_t, w)
			
			delta_w_t = alpha*(P_tp1-P_t)*e
			
			if t < len(sequence)-1:
				e = sequence[t+1] + l*e
			
			w += delta_w_t
	return w


def td(training_set, l, alpha):
	w = np.random.rand(5)
	w = np.ones(5)*0.5
	for sequence, outcome in training_set:
		e = sequence[0]
		delta_ws = []	
		for t, x_t in enumerate(sequence):
			if t == len(sequence)-1:
				P_tp1 = outcome
			else:
				x_tp1 = sequence[t+1]
				P_tp1 = np.dot(x_tp1, w)

			P_t = np.dot(x_t, w)
			
			delta_w_t = alpha*(P_tp1-P_t)*e
	    		delta_ws.append(delta_w_t)
			
			if t < len(sequence)-1:
				e = sequence[t+1] + l*e
			
		delta_w = np.sum(delta_ws, axis=0)
		w += delta_w
	return w


def td_repeated_presentations(training_set, l, alpha):
	w = np.random.rand(5)
	w = np.ones(5)*0.5
	for a in range(100000):
		delta_ws = []	
		for sequence, outcome in training_set:
			e = sequence[0]
			for t, x_t in enumerate(sequence):
				if t == len(sequence)-1:
					P_tp1 = outcome
				else:
					x_tp1 = sequence[t+1]
					P_tp1 = np.dot(x_tp1, w)

				P_t = np.dot(x_t, w)
				
				delta_w_t = alpha*(P_tp1-P_t)*e
		    		delta_ws.append(delta_w_t)
				
				if t < len(sequence)-1:
					e = sequence[t+1] + l*e
				
		delta_w = np.sum(delta_ws, axis=0)
		w += delta_w
 		if np.sum(np.abs(delta_w)) < alpha/10:
			return w
	return w

def brute_force(training_set):
	impressions = np.zeros(5)
	conversions = np.zeros(5)
	for sequence, outcome in training_set:
		for t, x_t in enumerate(sequence):
			impressions[np.where(x_t==1)]+=1
			if outcome ==1:
				conversions[np.where(x_t==1)]+=1
	return conversions/impressions

def expected_rewards():
	Q = np.matrix([[0,0.5,0,0,0],[0.5,0,0.5,0,0],[0,0.5,0,0.5,0],[0,0,0.5,0,0.5],[0,0,0,0.5,0]])
	I = np.identity(5)
	h = np.matrix([[0, 0, 0, 0, 0.5]]).T
	return np.linalg.inv(I-Q)*h

def expected_walk_lengths():
	Q = np.matrix([[0,0.5,0,0,0],[0.5,0,0.5,0,0],[0,0.5,0,0.5,0],[0,0,0.5,0,0.5],[0,0,0,0.5,0]])
	I = np.identity(5)
	return (np.linalg.inv(I-Q)*np.ones((5,1))).tolist()


expected_w = np.array([1/6., 1/3., 1/2., 2/3., 5/6.])
rand.seed(0)
np.random.seed(0)


training_sets = []
for _ in range(100):
	training_set = []
	for _ in range(100):
	    example = walk()
	    training_set.append(example)
	training_sets.append(training_set)


def fig_3():
	lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
	alpha = 0.001
	error_list = []
	for l in lambdas:
		errors = []
		for i, training_set in enumerate(training_sets):
			wo = td_repeated_presentations(training_set, l, alpha)
			error = math.sqrt(np.sum((wo-expected_w)**2)/5)
			print i, error
			errors.append(error)
		print l, np.mean(errors)
		error_list.append(np.mean(errors))

	plt.scatter(lambdas, error_list)
	plt.show()

def fig_4():
	lambdas = [0, 0.3, 0.8, 1]
	alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	for l in lambdas:
		error_list = []
		for alpha in alphas:
			errors = []
			for i, training_set in enumerate(training_sets):
				wo = td(training_set, l, alpha)
				error = math.sqrt(np.sum((wo-expected_w)**2)/5)
				errors.append(error)
			error_list.append(np.mean(errors))
		plt.plot(alphas, error_list)
	plt.show()

def fig_5():
	best=[]	
	lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
	alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	for l in lambdas:
		error_list = []
		for alpha in alphas:
			errors = []
			for i, training_set in enumerate(training_sets):
				wo = td(training_set, l, alpha)
				error = math.sqrt(np.sum((wo-expected_w)**2)/5)
				errors.append(error)
			error_list.append(np.mean(errors))

		best.append(np.min(error_list))

	plt.scatter(lambdas, best)
	plt.show()


def fig_intra_sequence():
	best=[]	
	lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
	alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	for l in lambdas:
		error_list = []
		for alpha in alphas:
			errors = []
			for i, training_set in enumerate(training_sets):
				wo = td_intra_sequence(training_set, l, alpha)
				error = math.sqrt(np.sum((wo-expected_w)**2)/5)
				errors.append(error)
			error_list.append(np.mean(errors))

		best.append(np.min(error_list))

	plt.scatter(lambdas, best)
	plt.show()



def fig_cumulative():
	expected = expected_walk_lengths()
	best=[]	
	lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
	alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
	for l in lambdas:
		error_list = []
		for alpha in alphas:
			errors = []
			for i, training_set in enumerate(training_sets):
				wo = td_cumulative(training_set, l, alpha)
				try:
					error = math.sqrt(np.sum((wo-expected)**2)/5)
				except ValueError:
					print wo-expected
				errors.append(error)
			error_list.append(np.mean(errors))

		best.append(np.min(error_list))

	plt.scatter(lambdas, best)
	plt.show()


fig_cumulative()
