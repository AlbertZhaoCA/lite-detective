import random
import os
import json
from utils.data_processing import remove_decorators_and_tags, get_all_files
from utils.templates import build_tieba_summary_prompt
from ..data_processing.preprocess import preprocess_csv
from utils.types import TieziSummary
from ..llm_sdk.llm import LLM

def getArgs():
    """
    Get the arguments for the policy generation.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Generate a policy for the data argumented agent to follow.')
    parser.add_argument('--path', type=str, default='data/raw', help='The path to get the data.')
    parser.add_argument('--policy_file', type=str, default='data/policy.jsonl', help='The path to the policy file.')
    parser.add_argument('--llm', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='The path to the llm (agent) model.')
    parser.add_argument('--type', type=str, default='remote', help='The type of the llm (agent) model.')
    parser.add_argument('--sample_times', type=int, default=5, help='The number of times to sample from the data.')
    parser.add_argument('--sample_size', type=int, default=5, help='The number of samples to generate from the data.')
    return parser.parse_args()

def generat_random_policy(llm,path: str = 'data/raw'):
    """Generate a random policy for the agent to follow. 
    """
    files = get_all_files(path, "csv")
    file = random.choice(files)
    data = preprocess_csv(file)
    data = random.choices(data,k=5)
    print(f"Generating policy for {file}..., {data}")
    processed_data = str(data)
    res = llm.generate(build_tieba_summary_prompt(processed_data))
    print(res)
    policy_file = 'data/policy.jsonl'
    with open(policy_file, 'a', encoding='utf-8') as f:
        res = json.loads(res)
        res_json = TieziSummary(**res).model_dump()
        json.dump(res_json, f, ensure_ascii=False)
        f.write('\n')

def _generate_policy(llm,file: str,saving_path: str = 'data/policy.jsonl',sample_times: int = 5,sample_size: int = 5):
    """
    Generate a policy for a csv file.
    Args:
        llm: The llm to use.
        file: The csv file to generate a policy for.
        saving_path: The path to save the policy.
    """
    for i in range(sample_times):
        data = preprocess_csv(file)
        data = random.choices(data,k=sample_size)
        processed_data = str(data)
        res = llm.generate(build_tieba_summary_prompt(processed_data))
        res = remove_decorators_and_tags(res)
        print(res)
        with open(saving_path, 'a', encoding='utf-8') as f:
            res = json.loads(res)
            res_json = TieziSummary(**res).model_dump()
            print(res_json)
            json.dump(res_json, f, ensure_ascii=False)
            f.write('\n')

def generate_policy(path: str = 'data/raw', policy_file: str = 'data/policy.jsonl', llm: str = 'Qwen/Qwen2.5-7B-Instruct', type: str = 'remote',sample_times: int = 5, sample_size: int = 5):
    """
    Generate a policy for the data argumented agent to follow.

    Args:
        path: The path to the data.
    """
    print("Generating policy 🚀")
    llm = LLM(os.getenv("AGENT_MODEL_PATH",llm),type=type)
    files = get_all_files(path, "csv")
    for file in files:
        try:
            print(f"Generating policy for {file} 🕵️")
            _generate_policy(llm,file,policy_file,sample_times,sample_size)
        except Exception as e:
            print(f"Error generating policy for {file}: {e}")
            continue

def main():
    """
    Main function for the policy generation.
    """
    args = getArgs()
    generate_policy(args.path, args.policy_file, args.llm, args.type, args.sample_times, args.sample_size)


if __name__ == "__main__":
    """for testing purposes"""
    generate_policy()
