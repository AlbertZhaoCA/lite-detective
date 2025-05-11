import json

def jsonl_to_json(input_file, output_file):
    with open("/root/llm-security/lite-detective/v2/data/train.jsonl", 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, 1):
            try:
                batch = json.loads(line)
                for i in batch:
                    with open(output_file, 'a', encoding='utf-8') as outfile:
                        json.dump(i, outfile, ensure_ascii=False)
                        outfile.write('\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue
            except Exception as e:
                print(f"An error occurred on line {line_number}: {e}")
# 使用示例
input_path = 'train.jsonl'
output_path = 'train2.json'
jsonl_to_json(input_path, output_path)
