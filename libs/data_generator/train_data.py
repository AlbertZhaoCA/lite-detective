import json
import os
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from libs.llm_sdk.llm import LLM
from utils.templates import build_training_data_prompt

def getArgs():
    parser = argparse.ArgumentParser(description='Generate training data for the model.')
    parser.add_argument('--policy_file', type=str, default='data/policy.jsonl')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--type', type=str, default='remote', choices=['remote', 'local'])
    parser.add_argument('--output_file', type=str, default='data/training_data.jsonl')
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=100)
    return parser.parse_args()

def load_policies(file_path="data/policy.jsonl"):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError:
                continue
    return data

def generate_one(policy: dict, llm_path: str, llm_type: str) -> dict | None:
    llm = LLM(os.getenv("AGENT_MODEL_PATH", llm_path), type=llm_type)
    try:
        res = llm.generate(build_training_data_prompt(policy['summary'], policy['policy']))
        return {"input": policy, "output": res}
    except Exception as e:
        print(f"[Error] {e}")
        return None
    finally:
        llm = None

def main():
    args = getArgs()
    print("Loading policies...")
    policies = load_policies(args.policy_file)
    print(f"Loaded {len(policies)} policies.")

    tasks = [random.choice(policies) for _ in range(args.num_samples)]
    output_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_processes) as executor, \
         tqdm(total=args.num_samples, desc="Generating training data 🚀") as bar, \
         open(args.output_file, 'a', encoding='utf-8') as fout:

        futures = {executor.submit(generate_one, policy, args.llm, args.type): policy for policy in tasks}

        for future in as_completed(futures):
            result = future.result()
            if result:
                with output_lock:
                    json.dump(result, fout, ensure_ascii=False)
                    fout.write('\n')
            bar.update(1)

    print("All samples generated and saved.")


if __name__ == "__main__":
    main()
