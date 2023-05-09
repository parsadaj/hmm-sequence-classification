import numpy as np
import subprocess


## Functions for train.py
def get_num_lines(path):
    """counts number of lines in the given file

    Args:
        path (string): path to file

    Returns:
        int: number of lines
    """
    return int(subprocess.check_output(['wc', '-l', path]).split()[0])


def get_states(n_states, states_path=None):
    if states_path is None:
        return np.arange(n_states), np.arange(n_states)
    return None, None


def read_model(init_model_path, sep=' '):
    model_file = open(init_model_path, 'r')
    lines = model_file.read().split('\n')
    
    n_states = int(lines[0].split()[1])
    
    Pi = np.fromstring(lines[1], sep=sep)
    A = np.fromstring(sep.join(lines[4:4+n_states]), sep=sep).reshape((n_states, n_states))
    B = np.fromstring(sep.join(lines[6+n_states:6+n_states+n_states]), sep=sep).reshape((n_states, n_states))
    
    model_file.close()
    
    return A, B, Pi


def test_read_init_model():
    a, b, pi = read_model('/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/hmm_data/model_init.txt', ' ')
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
    
    assert np.allclose(a, A)
    assert np.allclose(b, B)
    assert np.allclose(pi, Pi)
            
    
def write_model_to_file(A, B, Pi, out_path, sep=' '):
    model_file = open(out_path, 'w+')

    n_states = len(Pi)
    vocab_size = B.shape[0]
    
    model_file.write('initial: {}\n'.format(n_states))
    model_file.write(np.array2string(Pi, 1e100)[1:-1] + '\n\n')
    
    model_file.write('transition: {}\n'.format(n_states))
    for arr in A:
        model_file.write(np.array2string(arr, 1e100)[1:-1] + '\n')
        
    model_file.write('\nobservation: {}\n'.format(vocab_size))
    for arr in B:
        model_file.write(np.array2string(arr, 1e100)[1:-1] + '\n')
    
    model_file.close()
    

def test_write_model():
    A, B, Pi = read_model('/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/hmm_data/model_init.txt', ' ')
    write_model_to_file(A*1.324435345e-11, B*1.324435345e-11, Pi*1.324435345e-11, 'f.txt', sep=' ')
    

## Functions for test.py
def read_modellist(modellist_path):
    f = open(modellist_path, 'r')
    modellist = f.read().strip().split('\n')
    f.close()
    return modellist


def test_read_modellist():
    assert read_modellist("/Users/parsa/Daneshgah/Arshad/2/NLP/Homeworks/HW2/hmm_data/modellist.txt") == ["model_01.txt", "model_02.txt", "model_03.txt", "model_04.txt", "model_05.txt"]


def append_to_results(P, model_name, result_path, sep=' '):
    string_to_write = model_name + sep + str(P) + '\n'
    f = open(result_path, 'a+')
    f.write(string_to_write)
    f.close()
    

def append_to_log(P, log_path, sep='\n'):
    string_to_write = str(np.log(P)) + sep
    f = open(log_path, 'a+')
    f.write(string_to_write)
    f.close()

        
def test():
    pass
    # test_read_init_model()
    
    test_write_model()
    
    # test_read_modellist()


if __name__ == "__main__":
    test()
    