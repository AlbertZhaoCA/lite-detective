import pandas as pd
import json
import ast

# 1. 将已有的csv文件转化成特定格式的json文件
    # 格式: {"text": "科比的精神永远激励着我们", "label": 0, "context": ["科比的篮球精神让我们明白了坚持和努力的意义", "每次回顾他的故事都能重新找到前进的动力", "科比的名言一直激励着年轻一代去追逐梦想"]}
# 2. 支持将csv中内容加到已有的json文件中，使多个csv内容存在于一个json中

def csv_transfer_json(file_route, rows=None, write_mode='a'):
    df = pd.read_csv(file_route, encoding='utf-8-sig', skiprows=2, nrows=rows)
    texts = df['comment'].tolist()
    contexts = df['replys'].tolist()

    length = len(texts)
    print(f"共{length}条数据")
    for i in range(length):
        text = texts[i]
        context = contexts[i]
        context = ast.literal_eval(context) # 将"['自媒体做到最后，都是语不惊人死不休。']" 变为 ['自媒体做到最后，都是语不惊人死不休。']，字符串变列表

        # 每个元素的格式是字典
        data = {
            "text": text,
            "label": 0,
            "context":context
        }

        # 将字典写入json文件
        with open(r"data\val_real.jsonl", write_mode, encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=None)
            f.write("\n")

        if i % 5 == 0:
            print(f"\t已导入{i}条")

    print(f"共{length}条数据已全部导入json")

if __name__ == "__main__":
    file_route = r'D:\pycharm_code\pythonProject\Web_Crawler\data\回复：这就是你在你儿女眼中的样子【孙笑川的小家吧】_百度贴吧.csv'
    rows = 40 # 读取csv文件的前rows行 转入json文件
    write_mode = 'a' # json文件的写入是追加模式'a',  覆盖模式'w'
    csv_transfer_json(file_route, rows, write_mode)


