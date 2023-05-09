import os
import sys
import hmm
from utils import *


def main():
    args = sys.argv[1:]
    
    if len(args) != 4:
        raise Exception("Invalid system arguments!")
    # HW_path = '/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/'
    # O_path = os.path.join(HW_path, 'hmm_data/seq_model_01.txt')
    
    n_iterations, init_model_path, O_path, model_out_path = args
    
    log_path = os.path.join(os.path.dirname(model_out_path), 'log', os.path.basename(model_out_path) + '_PO.log')
    
    
    with open(log_path, 'w+') as f:
        f.write("")
    
    A, B, Pi = read_model(init_model_path, ' ')
    
    n_states = len(Pi)
    
    V = list("ABCDEF")
    v_to_k = {v:k for k,v in enumerate(V)}
    
    S, iS = get_states(n_states)
        
    new_A, new_B, new_Pi = hmm.baum_welch(int(n_iterations), O_path, iS, A, B, Pi, v_to_k, log_path)
    
    write_model_to_file(new_A, new_B, new_Pi, model_out_path)
        
if __name__ == '__main__':
    main()