from copy import deepcopy
import os
import numpy as np
from pyrsistent import v
import tqdm
import subprocess
from utils import *

np.seterr(all='raise')


def f(x, y, z, t):
    """calcuaresjfh fdsljfkhd

    Args:
        x (string): refers to sdnkfghjb
        y (int): _description_
        z (_type_): _description_
        t (_type_): _description_
    """


def baum_welch(n_iterations, O_path, iS, A, B, Pi, v_to_k, log_path='log/P_O.log'):
    """performs baum welch algorythm on all of the training data

    Args:
        n_iterations (int): after n_iteration iterations, algorythm will stop.
        O_path (path): path to training data file
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        v_to_k (dict): converts each word to its corresponding index
        
    Returns:
        tuple: final values for A, B, Pi
    """
    for _ in tqdm.tqdm(range(n_iterations)):
        A, B, Pi = baum_welch_iter(O_path, iS, A, B, Pi, v_to_k, log_path)
    return A, B, Pi

def baum_welch_iter(O_path, iS, A, B, Pi, v_to_k, log_path):
    """performs one iteration of baum welch algorythm on all of the training data

    Args:
        O_path (path): path to training data file
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        v_to_k (dict): converts each word to its corresponding index
        
    Returns:
        tuple: updated A, B, Pi
    """
    sigma_xi = np.zeros_like(A)
    sigma_gamma = np.zeros_like(Pi)
    sigma_gamma_k = np.zeros_like(B)
    sigma_gamma1 = np.zeros_like(Pi)
    
    counter = 0
    O = open(O_path, 'r')
    n_lines = get_num_lines(O_path)
    while True:
        O_line = O.readline()[:-1]
        
        if counter % 100 == 0:
            print(str(counter) + ' / ' + str(n_lines), end='\r')
        
        counter += 1
        
        if not O_line:
            break
        
        sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1, P_O = baum_welch_step(O_line, iS, A, B, Pi, v_to_k, sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1)
        
        append_to_log(P_O, log_path)
    O.close()
    
    A = sigma_xi / sigma_gamma[..., np.newaxis]
    B = sigma_gamma_k / sigma_gamma
    Pi = sigma_gamma1 / counter
    return A / np.sum(A, axis=1, keepdims=True), B / np.sum(B, axis=0, keepdims=True), Pi / np.sum(Pi, keepdims=True)


def baum_welch_step(O_line, iS, A, B, Pi, v_to_k, sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1):
    """performs baum welch algorythm on one line of training data

    Args:
        O_line (_type_): 1 line of training data
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        tuple: parameters used to update A, B, Pi
    """        
    alpha = get_alpha(iS, A, B, Pi, O_line, v_to_k)
    beta = get_beta(iS, A, B, Pi, O_line, v_to_k)

    K = np.array([v_to_k[o] for o in O_line])

    xi, gamma = compute_xi_gamma(alpha, beta, A, B, K)

    sigma_gamma1 =  sigma_gamma1 + gamma[0,...]
    sigma_gamma = np.sum(gamma, axis=0) + sigma_gamma
    sigma_xi = np.sum(xi, axis=0) + sigma_xi
    
    for k in range(sigma_gamma_k.shape[0]):
        sigma_gamma_k[k, ...] = sigma_gamma_k[k, ...] + np.sum(gamma[K==k], axis=0)
        
    P_O = np.sum(alpha[-1, ...])
    
    
    return sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1, P_O


def test_baum_welch():
    pass
    Pi = np.array([0.2,	0.1,	0.2,	0.2,	0.2,	0.1])
    A = np.fromstring('''0.3	0.3	0.1	0.1	0.1	0.1
0.1	0.3	0.3	0.1	0.1	0.1
0.1	0.1	0.3	0.3	0.1	0.1
0.1	0.1	0.1	0.3	0.3	0.1
0.1	0.1	0.1	0.1	0.3	0.3
0.3	0.1	0.1	0.1	0.1	0.3''', sep=' ').reshape((6,6))
    B = np.fromstring('''0.2	0.2	0.1	0.1	0.1	0.1
0.2	0.2	0.2	0.2	0.1	0.1
0.2	0.2	0.2	0.2	0.2	0.2
0.2	0.2	0.2	0.2	0.2	0.2
0.1	0.1	0.2	0.2	0.2	0.2
0.1	0.1	0.1	0.1	0.2	0.2''', sep=' ').reshape((6,6))
    
    V = ['A', 'B', 'C', 'D', 'E', 'F']
    v_to_k = {v:k for k,v in enumerate(V)}
    
    iS = np.arange(6)
    v_to_k = {v:k for k,v in enumerate(V)}
    k_to_v = {k:v for k,v in enumerate(V)}
    
    
    O = "ACCDDDDFFCCCCBCFFFCCCCCEDADCCAEFCCCACDDFFCCDDFFCCD\nCABACCAFCCFFCCCDFFCCCCCDFFCDDDDFCDDCCFCCCEFFCCCCBC\nABACCCDDCCCDDDDFBCCCCCDDAACFBCCBCCCCCCCFFFCCCCCDBF\nAAABBBCCFFBDCDDFFACDCDFCDDFFFFFCDFFFCCCDCFFFFCCCCD"

    sigma_xi = np.zeros_like(A)
    sigma_gamma = np.zeros_like(Pi)
    sigma_gamma_k = np.zeros_like(B)
    sigma_gamma1 = np.zeros_like(Pi)
    

    for o in O.split('\n'):
        sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1, p = baum_welch_step(o, iS, A, B, Pi, v_to_k, sigma_xi, sigma_gamma, sigma_gamma_k, sigma_gamma1)
        
    A = sigma_xi / sigma_gamma
    B = sigma_gamma_k / sigma_gamma
    Pi = sigma_gamma1 / 4
    return A, B, Pi

def compute_alpha(j, t, iS, A, B, Pi, O, alpha, v_to_k):
    """Calculates alpha 

    Args:
        j (int): current state index
        t (int): current time
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[k, s]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        alpha (2D array): contains computed alpha values (non computed values are filled with None)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        float: alpha(s, t) which is probability of being in state s on time t.
    """
    if t >= len(O):
        P_O = 0
        for i in iS:
            P_O += compute_alpha(i, t-1, iS, A, B, Pi, O, alpha, v_to_k)
        return P_O
    
    if alpha[t, j] is not None:
        return alpha[t, j]
    
    if t == 0:
        alpha[t, j] = Pi[j] * B[v_to_k[O[t-1]], j]
        return alpha[t, j]

    alpha_t_j = 0
    for i in iS:
        alpha_t_j += compute_alpha(i, t-1, iS, A, B, Pi, O, alpha, v_to_k) * A[i, j]
        
    alpha[t, j] = alpha_t_j * B[v_to_k[O[t]], j]
    return alpha[t, j]


def get_alpha(iS, A, B, Pi, O, v_to_k):
    """calculates and return a 2D array containing all alphas: alpha[t, i] is slpha at time t and state i

    Args:
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        dict: alpha
    """
    alpha = np.full((len(O), len(iS)), None)
    compute_alpha(0, 1+len(O), iS, A, B, Pi, O, alpha, v_to_k), 
    return alpha.astype(float)


def compute_beta(j, t, iS, A, B, Pi, O, beta, v_to_k):
    """Calculates beta 

    Args:
        j (int): current state index
        t (int): current time
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        beta (2D array): contains computed beta values (non computed values are filled with None)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        float: beta(s, t) which is probability of being in state s at time t on reversed string.
    """
    if t < 0:
        P_O = 0
        for i in iS:
            #compute_beta(i, 0, S, A, B, P, O, beta)
            P_O += Pi[i] * compute_beta(i, 0, iS, A, B, Pi, O, beta, v_to_k)
        return P_O
    
    if beta[t, j] is not None:
        return beta[t, j]
    
    if t == len(O) - 1:
        beta[t, j] = 1
        return beta[t, j]

    beta_j_t = 0
    for i in iS:
        beta_j_t += compute_beta(i, t+1, iS, A, B, Pi, O, beta, v_to_k) * A[j, i] * B[v_to_k[O[t+1]], i]
        
    beta[t, j] = beta_j_t
    return beta[t, j]


def get_beta(iS, A, B, Pi, O, v_to_k):
    """calculates and return a dict containing all betas

    Args:
        iS (list of ints): a list containing all state ineices
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        Pi (1D array): initial state (Pi[s] represents P(s | <START>))
        O (string): observations (O[t] is Observation at time t)
        v_to_k (dict): converts each word to its corresponding index

    Returns:
        dict: beta
    """
    beta = np.full((len(O), len(iS)), None)
    compute_beta(0, -1, iS, A, B, Pi, O, beta, v_to_k), 
    return beta.astype(float)


def test_alpha_beta():
    """A function to test if beta is calculated correctly using the example in lectures.

    Returns:
        dict: all betas
    """
    Pi = np.array([.85, .15])
    S = ['s', 't']
    iS = list(range(len(S)))
    V = ['A', 'B']
    
    v_to_k = {v:k for k,v in enumerate(V)}
    
    A = np.array([[.3, .7],
                  [.1,   .9]])

    B = np.array([[.4, .5],
                  [.6, 0.5]])
    
    O = 'ABBA'

    return get_alpha(iS, A, B, Pi, O, v_to_k), get_beta(iS, A, B, Pi, O, v_to_k)


def compute_xi_gamma(alpha, beta, A, B, K):
    """_summary_

    Args:
        alpha_t (2D array): alpha of current word
        beta_t (2D array): beta of current word
        beta_t1 (2D array): beta of next word
        A (2D array): transition matrix (A[s1, s2] represents P(s2 | s1) which is probability of transition from s1 to s2)
        B (2D array): observation matrix (B[s, k]] represents P(o | s) which is probability of observing o when being in state s and k is index of o in V(vocabulary list))
        o (observation_type): next word
        v_to_k (dict): converts each word to its corresponding index
        
    Returns:
        tuple: (xi, gamma)
    """    
    gamma = alpha * beta
    
    xi = alpha[:-1, :, np.newaxis] * A[np.newaxis, ...] * beta[1:, np.newaxis, ...] * B[K, ...][1:, np.newaxis, ...]
    
    return xi / np.sum(xi, axis=(1,2), keepdims=True), gamma / np.sum(gamma, axis=1, keepdims=True)


def test_xi_gamma(alpha, beta):
    """A function to test if xi and gamma are calculated correctly.

    Returns:
        None
    """
    Pi = np.array([.85, .15])
    S = ['s', 't']
    iS = list(range(len(S)))
    V = ['A', 'B']
    
    v_to_k = {v:k for k,v in enumerate(V)}
    
    A = np.array([[.3, .7],
                  [.1,   .9]])

    B = np.array([[.4, .5],
                  [.6, 0.5]])
    
    O = 'ABBA'
    K = np.array([v_to_k[o] for o in O])

    
    xi, gamma = compute_xi_gamma(alpha, beta, A, B, K)
    
    
    return xi, gamma


def get_best_model(line, modellist, output_path='output/'):
    """iterates through all models and computes P(line|model) for each and returns best model and its corresponding P

    Args:
        line (string): one observation
        modellist (list): list of model names
        output_path (str, optional): relative or absolute path to the folder containing models(i. e. model_0N.txt). Defaults to 'output/'.

    Returns:
        tuple: best_P, best_model
    """
    best_P = 0
    for model_name in modellist:
        model_path = os.path.join(output_path, model_name)
        P = eval_mdoel(model_path, line)
        if P > best_P:
            best_P = P
            best_model = model_name
    return best_P, best_model


def eval_mdoel(model_path, line):
    """evaluates given model on 1 observation and returns P(line|model)

    Args:
        mdoel_path (string): path/to/model_0N.txt
        line (string): 1 line of observation

    Returns:
        float: P(observation|model)
    """
    A, B, Pi = read_model(model_path, ' ')
    
    n_states = len(Pi)
    
    V = list("ABCDEF")
    v_to_k = {v:k for k,v in enumerate(V)}
    
    S, iS = get_states(n_states)
    
    alpha = get_alpha(iS, A, B, Pi, line, v_to_k)
    beta = get_beta(iS, A, B, Pi, line, v_to_k)
    
    P1 = np.sum(alpha[-1, ...])
    P2 = np.sum(beta[0, ...]*Pi)
    
    # try:
    #     assert np.allclose(np.array(P1), np.array(P2))
    # except AssertionError:
    #     print('********************** WARNING **********************')
    #     print("got different P(O|model) using forward and backward algorythms.")
    #     print('forward: {}'.format(P1))
    #     print('backward: {}'.format(P2))
    #     print('*****************************************************')

    return P1

def test_eval_model():
    Pi = np.array([1, 0])
    S = [0, 1]
    iS = list(range(len(S)))
    V = ['A', 'B']
    K =list(range(len(V)))
    
    v_to_k = {v:k for k,v in enumerate(V)}
    k_to_v = {k:v for k,v in enumerate(V)}
    
    A = np.array([[0.6, 0.4],
                  [0,   1  ]])

    B = np.array([[0.8, 0.3],
                  [0.2, 0.7]])
    
    O = 'AAB'
    
    n_states = len(Pi)
    
    alpha = get_alpha(iS, A, B, Pi, O, v_to_k)
    beta = get_beta(iS, A, B, Pi, O, v_to_k)
    
    P1 = np.sum(alpha[-1, ...])
    P2 = np.sum(beta[0, ...]*Pi)
    
    return P1, P2


def test_wikipedia():
    n_iterations, init_model_path, O_path, model_out_path = 4, '/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/wikipedia_data/model_init_wiki.txt', '/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/wikipedia_data/seq.txt', '/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/wikipedia_data/model_results.txt'

    A, B, Pi = read_model(init_model_path, ' ')

    n_states = len(Pi)

    V = list("AB")
    v_to_k = {v:k for k,v in enumerate(V)}

    S, iS = get_states(n_states)
        
    new_A, new_B, new_Pi = baum_welch(int(n_iterations), O_path, iS, A, B, Pi, v_to_k, log_path='/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/wikipedia_data/log/P.log')

    write_model_to_file(new_A, new_B, new_Pi, model_out_path)

 

def test():
    pass
    # print('---------------alpha-----------')
    # alpha, beta = test_alpha_beta()
    # print(alpha, type(alpha))
    # print('-------------------------------\n')

    # print('---------------beta-----------')
    # print(beta, type(beta)) 
    # print('-------------------------------\n')

    # print('---------------xi_gamma-----------')
    # xi, gamma = test_xi_gamma(alpha, beta)
    # for a in xi, gamma:
    #     pass
    #     # print(a.shape, end='\n\n')
    # # print('----------------------------------\n')

    # print('---------------baum_welch-----------')
    # b = test_baum_welch()
    # for B in b:
    #     print(B)
    # print('------------------------------------\n')
    
    # print('---------------eval-----------')
    # b, c = test_eval_model()
    # print(b, c)
    # print('------------------------------\n')
    test_wikipedia()
    

if __name__ == "__main__":
    test()
    
    