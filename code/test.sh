for test_num in 1 2
do
    echo testing on testing_data$test_num
    python3 code/test.py hmm_data/modellist.txt hmm_data/testing_data$test_num.txt output/result$test_num.txt
    echo done with testing_data$test_num
    echo ---------------------------------
done