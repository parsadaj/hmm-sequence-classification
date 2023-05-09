for model_file in $(cat hmm_data/modellist.txt)
do
    echo training on $model_file
    n_iter=10
    python3 code/train.py $n_iter hmm_data/model_init.txt hmm_data/seq_$model_file output/$model_file
    echo done with $model_file
    echo ---------------------------------
done