import pandas as pd
import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification

# 计算模型效果数据
def calculate_data(true_labels, prediction_labels):
    print("start calculating performance metrics")
    # 0表示正常，对于Negative
    # 1表示恶意，对于Positive
    TP = FP = TN = FN = 0
    for t, p in zip(true_labels, prediction_labels):
        if p == 1:
            if t == 1:
                TP += 1
            elif t == 0:
                FP += 1
        else:
            if t == 0:
                TN += 1
            elif t == 1:
                FN += 1
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    confu_matrix = f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}"

    # 计算accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 计算precision
    precision = TP / (TP + FP)
    # 计算recall
    recall = TP / (TP + FN)
    # 计算F1
    F1 = 2 * (precision * recall)/(precision + recall)
    print(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {F1}")
    performance_metrics = f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {F1}"
    return confu_matrix, performance_metrics

# 读取csv类型文本数据
def read_data(data_file):
    df = pd.read_csv(data_file, encoding="utf-8-sig", nrows=None) # nrows是读取数据的行数，不包括标题行。默认为None
    true_labels = df['label'].tolist()
    texts = df['TEXT'].tolist()
    return texts, true_labels

# 输出含模型预测结果的csv数据
def output(all_preds, all_labels, all_texts):
    data = {
        'prediction': all_preds,
        'labels': all_labels,
        'TEXT': all_texts
    }
    df = pd.DataFrame(data)
    df.to_csv('prediction_results.csv', mode = 'a', index=False,  encoding='utf-8-sig')


# 载入模型
tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold') # 分词器 将输入文本转化为模型看得懂的格式(例如张量)
model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold') # 载入已训练好的模型
model.eval()
print("已成功载入模型 Successfully load the model")

# 载入数据
# texts = ['你就是个傻逼！','黑人很多都好吃懒做，偷奸耍滑！','男女平等，黑人也很优秀。', '你他妈个畜生']
# true_labels = [coldet_baseline, coldet_baseline, 0, 0]
texts, true_labels = read_data("test_data.csv")

# 数据预测, 需要分批，否则容易在模型中向前运行时容易溢出
all_preds = []
all_labels = []
all_texts = []

print(f"开始数据预测，共{len(texts)}行数据 | Total {len(texts)} data")
batch_size = 32
for i in range(0, len(texts), batch_size):
    print(f"当前第{i}:{i+batch_size}行数据 | Current {i}:{i+batch_size} rows")
    batch_texts = texts[i:i+batch_size]
    batch_labels = true_labels[i:i+batch_size]
    model_input = tokenizer(batch_texts,return_tensors="pt",padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**model_input, return_dict=False)
    preds = torch.argmax(model_output[0].cpu(), dim=-1)
    preds= [p.item() for p in preds] # .item()获取pyTorch张量中的标量值

    # 将该论预测的数据存入了列表中
    all_preds.extend(preds)
    all_labels.extend(batch_labels)
    all_texts.extend(batch_texts)


# 1.计算预测结果性能 2.导出预测结果
# 计算
confusion_stats, performance_metrics = calculate_data(true_labels, all_preds)
# 将计算结果导入csv
with open('prediction_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
    f.write(confusion_stats + '\n')
    f.write(performance_metrics + '\n')
# 将数据导入csv
output(all_preds, all_labels, all_texts)





