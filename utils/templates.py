"""
LiteDetective - Malicious Content Detection Pipeline

Copyright (c) 2025 Albert Zhao
Author: Albert Zhao Zhaoq@kean.edu Hu
Created: 2025-05-11
Updated: 2025-05-11

Description:
    Templates for data generation and processing.

License: MIT License
"""

def build_tieba_summary_prompt(tiezi: str) -> str:
    """Build prompt for summarizing tieba posts.
    
    Args:
        tiezi (str): The tieba post.
        
    Returns:
        str: The formatted prompt.
    """
    return f"""你是一个帖子评审系统，你的任务是根据给定的评论区生成一套标准，用来挑选有害评论，并根据这些标准总结评论内容。
根据以下评论区内容，请总结出挑选有害评论的标准，包含哪些类型的评论应该被认为是有害的。例如，挑选标准可以包括：讽刺、侮辱、误导信息、恶意引导、负面情绪等。你需要给出每个标准的具体定义，并提供一到两个例子来展示如何判定有害内容。

帖子结构为:
[  {{
        'floor_number': number (评论的楼层),
        'comment': str(评论内容),
        'replys': list(评论的回复内容)
    }}]

评论: {tiezi}

根据这些评论，输出一套简洁的挑选标准，解释如何挑选出有害评论，并提供一些例子。
输出示例:
{{  'summary': '这是涉及建政的评论，包含了对社交网红的讽刺和侮辱。评论需要结合背景进行分析。',
    'policy': [
        {{'type': '讽刺', 'definition': '通过讽刺或嘲笑他人或某个话题来表现敌意。', 'examples': ['评论示例1', '评论示例2']}},
        {{'type': '侮辱', 'definition': '恶意攻击个人、群体或某种观点，带有明显的侮辱性语言。', 'examples': ['评论示例3', '评论示例4']}},
        {{'type': '误导信息', 'definition': '传播不实或夸大的信息，特别是涉及公共政策或科学内容时。', 'examples': ['评论示例5', '评论示例6']}},
        {{'type': '恶意引导', 'definition': '评论试图引发无意义的争论或恶性话题。', 'examples': ['评论示例7', '评论示例8']}},
        {{'type': '负面情绪', 'definition': '评论表达过度的负面情绪，且没有提出任何实质性的解决办法。', 'examples': ['评论示例9', '评论示例10']}}
    ]
}}
请只输出json格式的结果，包含每个标准的类型、定义和示例。请确保输出的JSON格式正确，不要进行解释，不要输出'''json'''。
"""

def build_training_data_prompt(summary: str, policy: list) -> str:
    """Build prompt for generating training data.
    
    Args:
        summary (str): Summary of the content.
        policy (list): List of policy definitions.
        
    Returns:
        str: The formatted prompt.
    """
    policy_text = "\n".join(
        f"类型: {p['type']}, 定义: {p['definition']}, 示例: {p['examples']}"
        for p in policy
    )

    return f"""你是一个文本分类模型的训练数据生成器。根据以下有害文本的类型定义生成训练数据：
总结: {summary}

有害文本: {policy_text}


通过有害文本的定义，以及有害文本的例子，生成新的评论 (text)，并标注其是否有害 (label: 0 表示无害，1 表示有害)。
你必须包含上下文，上下文 (context)是与评论相关的其他评论。
你必须构造几个类似的无害评论以及有害评论，并且每个评论都要尽量包含上下文。
请输出一个包含以下字段的 JSON 数组对象：

[{{ 'text': str(评论内容),'label': int(0或1),'context': str(其他相关评论) }}]

示例：[
{{'text': '我今天真开心', 'label': 0, 'context': '嗨小王，今天你过得怎么样？'}},
{{'text': '就知道上课，上尼玛呢', 'label': 1, 'context': '通知通知📢，五一不放假'}},
{{"text": "专科生也配评论技术？先去考个本科吧","label": 1,"context": "讨论：如何评价最新的AI框架？"}},
{{"text": "国家养老项目最后3个名额，投资5万每月返1万，速联李主任：138XXXX1234","label": 1,"context": "有什么靠谱的养老投资项目？"}},
{{"text": "出轨男车牌号京A12345，每天19点停国贸地下车库","label": 1,"context": "用户发帖：发现老公疑似出轨怎么办？"}}]

请确保输出的JSON数组格式正确，不要进行解释，不要输出'''json'''。
"""

def build_data_augmentation_prompt(text: str) -> str:
    """Build prompt for data augmentation.
    
    Args:
        text (str): The input text to augment.
        
    Returns:
        str: The formatted prompt.
    """
    return f"""你是一个用于构建有害内容识别数据集的文本生成器。

请根据用户的发言（text），构造出相关的对话背景（context，即其他人的发言），并以合法 JSON 格式输出。

任务目标是模拟出用户的 text 出现在某个对话背景中，并判断该 text 在该 context 下，是否构成有害内容（label）。

【重要说明】：
- 判断对象始终是 text；
- 需要结合 context 语境来判断 text 是否构成有害言论；
- 不是单独判断 text，也不是把 context + text 当作一个整体来判断；
- 要判断的是：**在该 context 背景下，text 是否构成攻击性、冒犯、歧视、误导、煽动、欺诈等倾向**。

label 含义说明：
- label = 1 表示：在给定 context 中，text 构成有害言论；
- label = 0 表示：在给定 context 中，text 属于中立、理性或健康讨论；

其他要求：
1. context 至少包含 2 条发言，能自然引出 text；
2. 若 text 无法支持给定 label，请合理切换 label 并重新构造 context；
3. 输出必须为合法 JSON，结构如下：
   {{
     "text": "...",
     "label": 0 or 1,
     "context": ["...", "...", "..."]
   }}

示例：

1. 输入：
   text: "你脑子有问题吧？这种破事也拿出来说！"
   输出：
   {{
     "text": "你脑子有问题吧？这种破事也拿出来说！",
     "label": 1,
     "context": [
       "我只是分享了一下我今天遇到的麻烦事。",
       "每个人都有情绪，不应该被这样对待。",
       "他可能只是想找人倾诉一下。"
     ]
   }}

2. 输入：
   text: "我觉得这个提案蛮不错的，值得尝试。"
   输出：
   {{
     "text": "我觉得这个提案蛮不错的，值得尝试。",
     "label": 0,
     "context": [
       "我们目前在讨论是否使用新的策略。",
       "这个提案有点冒险，不过也可能带来新机会。",
       "大家觉得这个方案如何？"
     ]
   }}

输入：
text: "{text}"

请输出构造的 context，以 JSON 格式返回，不要输出其他内容，也不要加代码标记。
"""