import json
from libs.llm_sdk.llm import LLM
from tqdm import tqdm
from utils.templates import build_data_inspection_prompt
from utils.data_processing import read_jsonl, remove_decorators_and_tags
from utils.types import TrainingDataItem
from utils.types import EvaluationDataItem

def getArgs():
    """
    Get the arguments for the data inspection.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the training data.')
    parser.add_argument('--path', type=str, default='data/training_data.jsonl', help='The path to the training data.')
    parser.add_argument('--model', type=str, default='Qwen3-32B-AWQ', help='The path to the llm (agent) model.')
    parser.add_argument('--type', type=str, default='remote', help='The type of the llm (agent) model.')
    return parser.parse_args()

def inspect_training_data(path:str,model:str,model_type:str):
    """
    Inspect the training data, leave highly confidence data in the file
    Extract the suspected wrong labeled data out of training data. Correct the 
    labels and save them in a new file.
    
    Args:
        data (str): The data path to inspect.
        model (str): The model to use for inspection.
        model_type (str): The type of the model to use for inspection.
    
    Returns:
        None
    """
    data = read_jsonl(path)
    print("Data Inspection 🔍:")
    print(f"Number of records: {len(data)}")
    llm = LLM(model, type=model_type)
    for record in tqdm(data, desc="Inspecting records", unit="record"):
        try:
            if 'label' in record:
                res = llm.generate(build_data_inspection_prompt(record['text'], record['context'], [record['label']]))
                res = EvaluationDataItem(**json.loads(remove_decorators_and_tags(res)))
                if res.label == record['label']:
                    with open(f"{path}_correct_verified.jsonl", "a", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
                else:
                    res = res.model_dump()
                    with open(f"{path}_incorrect_verified.jsonl", "a", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False)
                        f.write('\n')
            elif 'label_without_context' in record:
                res = llm.generate(build_data_inspection_prompt(record['text'], record["context"], [record['label_without_context'], record['context']] ))
                res = TrainingDataItem(**json.loads(remove_decorators_and_tags(res)))
                if res.label_without_context == record['label_without_context'] and res.label_with_context == record['label_with_context']:
                    with open(f"{path}_correct_verified.jsonl", "a", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
                else:
                    res = res.model_dump()
                    with open(f"{path}_incorrect_verified.jsonl", "a", encoding="utf-8") as f:
                        json.dump(res, f, ensure_ascii=False)
                        f.write('\n')
        except Exception as e:
            print(f"[Error] {e}")
    else:
        print("No records found.")
    print("-" * 40)

def main():
    args = getArgs()
    inspect_training_data(args.path, args.model, args.type)



