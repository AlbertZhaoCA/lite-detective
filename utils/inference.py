from transformers import BertTokenizer

def get_tokenizer(bert_name='bert-base-chinese'):
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    return tokenizer