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
import random
import json

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

分析评论内容，输出以下两个字段：

1. "summary": 对整体评论的主题和有害内容特征的简要总结（不少于10个字）。
2. "policy": 有害评论识别标准的列表，每个标准需包含：
   - type: 标准类型（如"人身攻击"、"误导信息"等）
   - definition: 该类型的定义，用简洁客观的语言描述
   - examples: 一到两个实际评论片段作为判定示例

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
    context_len = 4 if random.random() > 0.05 else 6

    richness_hint = f"context 评论数量约为 {context_len} 条左右，每条的长短不一。"
    examples = [
        {
            "text": "你脑子有问题吧？这种破事也拿出来说！",
            "label_without_context": 1,
            "label_with_context": 1,
            "context": [
                "我只是分享了一下我今天遇到的麻烦事。",
                "每个人都有情绪，不应该被这样对待。",
                "他可能只是想找人倾诉一下。"
            ]
        },
        {
            "text": "我觉得这个提案蛮不错的，值得尝试。",
            "label_without_context": 0,
            "label_with_context": 0,
            "context": [
                "我们目前在讨论是否使用新的策略。",
                "这个提案有点冒险，不过也可能带来新机会。",
                "大家觉得这个方案如何？"
            ]
        }, {
            "text": "你这是在浪费大家时间，根本没有意义！",
            "label_without_context": 1,
            "label_with_context": 0,
            "context": [
                "我提议的方案可能不是最好的，但我们应该听听大家的意见。",
                "每个人都有不同的看法，不能一开始就否定。",
                "这不是浪费时间，而是在讨论如何改进问题。"
            ]
        },{
        "text": "你这是在故意挑拨离间吧？！",
        "label_without_context": 1,
        "label_with_context": 0,
        "context": [
            "我只是想提醒大家注意一些问题，希望大家能一起解决。",
            "我觉得他可能只是想引起大家的注意，并没有恶意。",
            "我们不应该轻易误解别人，应该多沟通。"
        ]
            },{
            "text": "你是不是故意找茬啊？！",
            "label_without_context": 1,
            "label_with_context": 0,
            "context": [
                "我只是想指出问题，希望能改进。",
                "我觉得他可能只是想表达自己的想法，并没有恶意。",
                "我们不应该轻易误解别人，应该多沟通。"
            ]
        },{
            "text": "你这是在胡说八道吧？！",
            "label_without_context": 1,
            "label_with_context": 0,
            "context": [
                "我只是想分享我的想法，也许能带来新的思路。",
                "我觉得他的想法虽然有点离奇，但也挺有趣的。",
                "我们不应该轻易否定别人的想法。"
            ]
        }
    ]

    sampled_examples = random.sample(examples, k=2)
    example_text = json.dumps(sampled_examples, ensure_ascii=False, indent=2)
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

    return f"""你是一个文本分类模型的训练数据生成器。

任务目标：
我们要识别一条评论 `text` 是否构成有害内容。请注意，判断分为两种情况：
1. `label_without_context`：仅考虑 `text` 自身内容是否有害；
2. `label_with_context`：结合 `context`（即前文评论）判断 `text` 是否在语境下有害。

以下是你需要生成的有害内容主题与类型定义：

聊天区内容主题：
{summary}

有害内容类型及示例：
{policy_text}

生成要求：
1. 每条数据包括一条评论 `text`，一段 `context`（由多条与之相关的、时间上在前的评论组成），以及两个标签：
   - `label_without_context`: 0（无害）或 1（有害），基于 text 本身；
   - `label_with_context`: 0（无害）或 1（有害），考虑 context 情况下的判断。

2. 格式如下（注意字段顺序）：
{{
  "text": "评论内容",
  "label_without_context": 0 或 1,
  "label_with_context": 0 或 1,
  "context": ["该评论前面第1条评论", "该评论前面第二条评论","该评论前面第三条评论","该评论前面第四条评论" ...]
}}

3. 注意事项：
   - `text` 你需要构造的评论；
   - `context` 你需要构造的评论聊天区对话；
   - `context` 仅包含在 `text` 之前出现的评论；
   - `context` 应该是真实评论，而非对话背景的描述；
   - {richness_hint}
   - 生成有害与无害的样本应平衡；
   - 输出为 JSON 数组对象；
   - 请确保语义自然，贴近真实评论场景；
   - 不要输出任何说明或 markdown 标签（如 ```json）；
   - 保证输出的 JSON 严格合法。

4. 示例：
{example_text}

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

def build_data_inspection_prompt(text: str, context:list=None, label:list=None) -> str:
    """Build prompt for data inspection.
    
    Args:
        text (str): The input text to inspect.
        context (list): The context of the text.
        label (list): The labels for the text.
            - label[0] is for label_without_context
            - label[1] is for label_with_context.
        
    Returns:
        str: The formatted prompt.
    """
    
    if len(label) == 1:
        return f"""你是一个文本分类模型的评估助手。
请根据用户的发言，判断其是否构成有害内容，并以合法 JSON 格式输出。
任务目标是判断该 检测文本 是否构成攻击性、冒犯、歧视、误导、煽动、欺诈等倾向。
label 含义说明：
- label = 1 表示：该 text 构成有害言论；
- label = 0 表示：该 text 属于中立、理性或健康讨论；
其他要求：
1. 输出必须为合法 JSON，结构如下：
   {{
     "text": "...",
     "label": 0 or 1,
     "context": []
   }}
示例：
1. 输入：
   检测文本: "你脑子有问题吧？这种破事也拿出来说！"
   标签: 1
   上文背景: ["傻逼", "你脑子有问题吧？这种破事也拿出来说！"]
   输出：
   {{
     "text": "你脑子有问题吧？这种破事也拿出来说！",
     "label": 1,
     "context":["傻逼", "你脑子有问题吧？这种破事也拿出来说！"]
   }}
2. 输入：
   检测文本: "我觉得这个提案蛮不错的，值得尝试。"
   标签: 1
   上文背景: ["谁写的💩玩意。","写的这吗差吗，服了","其实也还好"]
   输出：
   {{
     "text": "我觉得这个提案蛮不错的，值得尝试。",
     "label": 0,
      context: ["谁写的💩玩意。","写的这吗差吗，服了","其实也还好"]
   }}

用户输入：
检测文本: {text}
标签: {label[0]}
上文背景: {context}
"""
    elif len(label) == 2:
        return f"""你是一个文本分类模型的评估助手。
请根据用户的发言（text）和评论区背景（context），判断该 text 是否构成有害内容，并以合法 JSON 格式输出。
任务目标是判断该 text 在此评论区下是否构成攻击性、冒犯、歧视、误导、煽动、欺诈等倾向。
label 含义说明：
- label = 1 表示：在给定 context 中，text 构成有害言论；
- label = 0 表示：在给定 context 中，text 属于中立、理性或健康讨论；
其他要求：
1. 输出必须为合法 JSON，结构如下：
   {{
     "text": "...",
     "label_with_context": 0 or 1,
     "label_without_context": 0 or 1,
     "context": ["...", "...", "..."]
   }}
示例：
1. 输入：
   text: "你脑子有问题吧？这种破事也拿出来说！"
   label: 1
   context: ["傻逼", "你脑子有问题吧？这种破事也拿出来说！"]
   输出：
   {{
     "text": "你脑子有问题吧？这种破事也拿出来说！",
     "label_with_context": 1,
     "label_without_context": 1,
     "context":["傻逼", "你脑子有问题吧？这种破事也拿出来说！"]
   }}
2. 输入：
   text: "我觉得这个提案蛮不错的，值得尝试。"
   label: 1
   context: ["谁写的💩玩意。","写的这吗差吗，服了","其实也还好"]
   输出：
   {{
     "text": "我觉得这个提案蛮不错的，值得尝试。",
     "label_with_context": 0,
     "label_without_context": 1,
     "context": ["谁写的💩玩意。","写的这吗差吗，服了","其实也还好"]
   }}
用户输入：
text: {text}
label_without_context: {label[0]}
label_with_context: {label[1]}
context: {context}
"""