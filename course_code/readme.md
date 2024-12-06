## How to run

Checkout to branch songhang_distance_sum if you want to test distance function = inverse distance sum, songhang_softmax if you want to test distance function = softmax

Please clear `data/knn_datastore` if you are switching from one dataset from another dataset, our code will rebuild the KNN datastore.

In `course_code` folder, run `python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "knn_LM" --llm_name "meta-llama/Llama-3.2-3B-Instruct" --split 1`

To adjust `lambda`, you can just modify `LAMBDA_KNN` in `course_code/knn_lm.py`. The default value is 0.1.

To adjust `unsure_bar`, you can just modify `UNSURE_BAR` in `course_code/knn_lm.py`. The default value is 0.6.

If `lamdbda=0`, you can run `base_lm` to get a result faster, the command is `python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "base_LM" --llm_name "meta-llama/Llama-3.2-3B-Instruct" -- split 1`

### How to evaluate

Just run `python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "knn_LM" --llm_name "meta-llama/Llama-3.2-3B-Instruct"` or `python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "base_LM" --llm_name "meta-llama/Llama-3.2-3B-Instruct"`, depends on what model you want.

Currently in `output` folder, the result for `knn_LM` used unsure_bar = 0.6, lambda = 0.1, distance function = branch name(inversed distance sum in master branch). The code to save result only allows one prediction file for one model and dataset, so we don't have the prediction file under other parameters in the output directory. To reproduce them, just modify the parameter as "How to run" part mentioned and run the generation.