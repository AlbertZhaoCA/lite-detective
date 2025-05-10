import random
import os
import json
from utils.data_processing import remove_json_decorators, get_all_files, load_csv
from utils.templates import build_tieba_summary_prompt
from libs.data_processing.preprocess import preprocess_csv
from models.data import TieziSummary
from ..llm_sdk.local_llm import LLM



def generat_random_policy(llm,path: str = 'data/raw'):
    """
    Generate a random policy for the agent to follow.
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

def _generate_policy(llm,file: str):
    """
    Generate a policy for a csv file.
    """
    data = preprocess_csv(file)
    processed_data = str(data)
    res = llm.generate(build_tieba_summary_prompt(processed_data))
    res = remove_json_decorators(res)

    policy_file = 'data/policy.jsonl'
    with open(policy_file, 'a', encoding='utf-8') as f:
        res = json.loads(res)
        res_json = TieziSummary(**res).model_dump()
        json.dump(res_json, f, ensure_ascii=False, indent=4)
        f.write('\n')

def generate_policy(path: str = 'data/raw'):
    """
    Generate a policy for the data argumented agent to follow.

    Args:
        path: The path to the data.
    """
    print("Generating policy...")
    llm = LLM(os.getenv("AGENT_MODEL_PATH"))
    files = get_all_files(path, "csv")
    for file in files:
        try:
            print(f"Generating policy for {file}...")
            _generate_policy(llm,file)
        except Exception as e:
            print(f"Error generating policy for {file}: {e}")
            continue


if __name__ == "__main__":
    generate_policy()
