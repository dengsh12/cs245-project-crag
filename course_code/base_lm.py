import os
import time
import faiss
import numpy as np
from typing import Any, Dict, List
import torch
import vllm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vanilla_baseline import VLLM_MAX_MODEL_LEN
from tqdm.auto import tqdm
#### CONFIG PARAMETERS ---

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.85 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
DATASTORE_PATH = "../data/knn_datastore"
LAMBDA_KNN = 0
KNN_NEIGHBOUR_NUM = 10
UNSURE_BAR = 0.7

class BaseLM:
    def __init__(self, llm_name="meta-llama/Llama-3.2-3B-Instruct", is_server=False, vllm_server=None, dataset_path = ""):
        """
        Initialize your model(s) here if necessary.
        This is the constructor for your DummyModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        self.initialize_models(llm_name, is_server, vllm_server, dataset_path)

    def initialize_models(self, llm_name, is_server, vllm_server, dataset_path):
        self.llm_name = llm_name
        self.is_server = is_server
        self.vllm_server = vllm_server
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        index_path = os.path.join(DATASTORE_PATH, "datastore.index")
        tokens_path = os.path.join(DATASTORE_PATH, "tokens.npy")
        hidden_dim = self.model.config.hidden_size
        self.avg_prob_list = []


    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE  
        return self.batch_size

    def batch_generate_answer(self, batch: Dict[str, Any]) -> List[str]:
        """
        Generates answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters:
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                      to a query. Please refer to the following link for
                                                      more details about the individual search objects:
                                                      https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/blob/master/docs/dataset.md#search-results-detail
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.

        Returns:
            List[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes:
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        batch_interaction_ids = batch.get("interaction_id", [])
        queries = batch.get("query", [])
        query_times = batch.get("query_time", [])

        answers = []

        for query, query_time in zip(queries, query_times):
            context = self.build_context(query, query_time)
            # try:
                
            # except Exception as e:
            #     print(e)
            #     answer = "I don't know"
            answer = self.generate_answer(
                    context=context,
                    max_tokens=50,
                    uncertainty_threshold=0.5
                )
            answers.append(answer)

        return answers
    
    def build_datastore(self):
        from generate import load_data_in_batches
        # vLLM doesn't support extracting hidden state, to get the hidden state, we go this way.
        dataset_path = self.dataset_path
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(self.llm_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id not in tokenizer.all_special_ids:
            tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})
            model.resize_token_embeddings(len(tokenizer)) 
            
        tokenizer.model_max_length = VLLM_MAX_MODEL_LEN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        hidden_dim = model.config.hidden_size
        embeddings = []
        tokens = []
        batch_size = self.get_batch_size()
        index = faiss.IndexFlatL2(hidden_dim)
        special_ids = set(tokenizer.all_special_ids)
        assert tokenizer.pad_token_id in special_ids, "pad_token_id should be in tokenizer.all_special_ids"

        filtered_count = 0
        total_tokens = 0
        load_start_t = time.time()
        for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Building dataStore"):
            queries = batch["query"]
            inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            for i, query in enumerate(batch["query"]):
                input_ids = inputs.input_ids[i]
                hidden = hidden_states[i]
                for j in range(1, len(input_ids)):
                    current_token_id = input_ids[j].item()
                    # prev_token_id = input_ids[j-1].item()
                    if current_token_id not in special_ids:
                        embeddings.append(hidden[j-1].cpu().numpy())
                        tokens.append(current_token_id)
                    else:
                        filtered_count += 1
                    total_tokens += 1

        load_end_t = time.time()
        print(f"Total tokens: {total_tokens}, Filtered tokens: {filtered_count}")
        print(f"Used time:{load_end_t - load_start_t}")
        embeddings = np.array(embeddings, dtype="float32")
        tokens = np.array(tokens, dtype="int64")
        index.add(embeddings)
        faiss.write_index(index, f"{DATASTORE_PATH}/datastore.index")
        np.save(f"{DATASTORE_PATH}/tokens.npy", tokens) 
        self.avg_prob_list = []

    def build_context(self, query: str, query_time: str) -> str:
        system_prompt = "You are provided with a question. Your task is to answer the question succinctly, using the fewest words possible."
        context = f"{system_prompt}\nQuestion: {query}"
        return context

    def generate_answer(self, context: str, max_tokens: int = 75, uncertainty_threshold: float = 0.5) -> str:
        """

        """
        input_ids = self.tokenizer.encode(context, return_tensors='pt').to(self.device)
        generated_ids = input_ids.tolist()[0]
        unsure_cnt = 0
        total_cnt = 0
        initial_length = len(generated_ids)
        avg_prob = 0
        for _ in range(max_tokens):
            total_cnt += 1
            # Get context embeddings
            outputs = None
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # shape: (1, seq_length, hidden_size)
                context_embedding = hidden_states[:, -1, :].squeeze(0).cpu().numpy()  # shape: (hidden_size,)

            with torch.no_grad():
                lm_logits = outputs.logits[:, -1, :] 
                lm_probs = torch.softmax(lm_logits, dim=-1).cpu().numpy().squeeze(0) 
                max_lm_prob = lm_probs.max()

            # Bombine KNN and LM
            combined_probs = lm_probs
            combined_probs /= combined_probs.sum() 

            max_prob = combined_probs.max()
            avg_prob += max_prob
            
            # if max_prob < uncertainty_threshold:
            #     unsure_cnt += 1
            #     if len(generated_ids) > len(input_ids[0]):
            #         break
            #     else:
            #         return "I don't know"
                

            next_token_id = np.argmax(combined_probs)
            generated_ids.append(next_token_id)
            input_ids = torch.tensor([generated_ids], device=self.device)

        avg_prob /= max_tokens
        self.avg_prob_list.append(avg_prob)
        generated_text = self.tokenizer.decode(generated_ids[initial_length:], skip_special_tokens=True)

        tokens = self.tokenizer.tokenize(generated_text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            generated_text = self.tokenizer.convert_tokens_to_string(tokens)

        print(f'batch:unsure_cnt={unsure_cnt}, total_cnt={total_cnt}')
        if not generated_text.strip():
            return "I don't know"

        if avg_prob < UNSURE_BAR:
            return "I don't know"

        return generated_text