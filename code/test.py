# import os
import sys
import hmm
from utils import *

def main():
    args = sys.argv[1:]
    
    if len(args) != 3:
        raise Exception("Invalid system arguments!")
    # HW_path = '/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/'
    # O_path = os.path.join(HW_path, 'hmm_data/seq_model_01.txt')
    
    modellist_path, test_data_path, result_out_path = args
    
    with open(result_out_path, 'w+') as f:
        f.write("")
    
    modellist = read_modellist(modellist_path)
    
    test_data_file = open(test_data_path, 'r')

    counter = 0
    n_lines = get_num_lines(test_data_path)
    
    while True:
        test_line = test_data_file.readline()[:-1]
        
        if counter % 100 == 0:
            print(str(counter) + ' / ' + str(n_lines), end='\r')
        
        counter += 1
        
        if not test_line:
            break
        P, model_name = hmm.get_best_model(test_line, modellist)
        append_to_results(P, model_name, result_out_path)
        
    
    
if __name__ == '__main__':
    main()