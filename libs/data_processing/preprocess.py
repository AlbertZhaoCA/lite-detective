import csv
from utils.data_processing import clean_data

def preprocess_csv(csv_file):
    result = []
    
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        cleaned_fieldnames = [clean_data(field) for field in reader.fieldnames]
        
        reader.fieldnames = cleaned_fieldnames
        
        for row in reader:
            result.append(row)
    
    return result

if __name__ == "__main__":
    """for testing purposes"""
    from ..llm_sdk.local_llm import LLM
    import os
    import json
    from utils.data_processing import remove_decorators_and_tags
    from utils.templates import build_tieba_summary_prompt
    csv_file = 'data/raw/(10 封私信 _ 83 条消息) 为什么户晨风会如此狂烈地支持私有制医疗？ - 知乎.csv'
    data = preprocess_csv(csv_file)
    processed_data = str(data[:5])
    
    llm = LLM(os.getenv("AGENT_MODEL_PATH"))
    res = llm.generate(build_tieba_summary_prompt(processed_data))
    print("before",res)
    res = remove_decorators_and_tags(res)
    print("after",res)

    with open('data/policy.json', 'w', encoding='utf-8') as f:
        res_json = json.loads(res)
        json.dump(res_json, f, ensure_ascii=False, indent=4)
   
