import os
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 模型目录路径
model_dir = r"./results"

# 获取所有 checkpoint 文件夹
checkpoints = [d for d in os.listdir(model_dir) if re.match(r'checkpoint-\d+', d)]
if not checkpoints:
    print("没有找到任何 checkpoint 文件夹")
    exit()

# 按照数字排序选择最新的 checkpoint
checkpoints.sort(key=lambda x: int(x.split('-')[1]))

# 选择最新的 checkpoint 目录
latest_checkpoint = checkpoints[-1]
model_path = os.path.join(model_dir, latest_checkpoint)
print(f"加载最新的 checkpoint: {model_path}")

try:
    # 加载预训练的 BERT 分词器（假设使用的是 'bert-base-chinese'）
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # 加载训练好的模型
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    print("模型和分词器加载成功")
except Exception as e:
    print(f"加载模型或分词器时出错: {e}")
    exit()

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def classify_comment(comment):
    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id

def classify_comments(comments):
    results = []
    for comment in comments:
        result = classify_comment(comment)
        results.append(result)
    return results

# 测试评论
test_comments = ["你妈死了", "今天天气挺好的哈哈", "我稀饭你"]
results = classify_comments(test_comments)
for comment, result in zip(test_comments, results):
    print(f"评论: {comment} -> 分类结果: {'脏话' if result == 1 else '正常'}")
