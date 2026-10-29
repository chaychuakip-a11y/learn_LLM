"""用原生生成前缀和原生结束符序列化；不需要导入torch。"""
def encode_example(tokenizer, prompt, completion, max_length):
    if len(completion) != 1 or completion[0].get('role') != 'assistant':
        raise ValueError('E_COMPLETION: V1每步需要唯一assistant答案')
    answer = completion[0].get('content')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('E_EMPTY_ANSWER: 答案为空')
    options = {'tokenize': False, 'enable_thinking': False}
    base = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, **options)
    generation_prefix = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, **options)
    full_native = tokenizer.apply_chat_template(prompt+completion, add_generation_prompt=False, **options)
    if not full_native.startswith(base):
        raise ValueError('E_TEMPLATE: 原生完整对话不以用户对话为前缀，需要适配模板')
    response = full_native[len(base):]
    if response.count(answer) != 1:
        raise ValueError('E_TEMPLATE: 原生模板改写或重复了答案，不能猜测loss边界')
    suffix = response.split(answer, 1)[1]
    if not suffix or len(suffix) > 256:
        raise ValueError('E_TEMPLATE_END: 未识别到简短的原生assistant结束符')
    # Gemma非思考推理前缀含空thought块，而完整对话渲染可能不含；
    # 用真实推理前缀+原始答案+原生结束符，保证训练与推理一致。
    prefix_ids = tokenizer.encode(generation_prefix, add_special_tokens=False)
    target_ids = tokenizer.encode(answer+suffix, add_special_tokens=False)
    full = prefix_ids + target_ids
    if tokenizer.encode(generation_prefix + answer + suffix, add_special_tokens=False) != full:
        raise ValueError('E_TOKEN_BOUNDARY: 模板边界发生分词合并，不能猜测答案掩码')
    if len(full) > max_length:
        raise ValueError(f'E_LENGTH: {len(full)} tokens > {max_length}；提炼证据或分步，不静默截断')
    if not target_ids:
        raise ValueError('E_LOSS_MASK: 没有可学习的目标token')
    return {'input_ids': full, 'attention_mask': [1]*len(full),
            'labels': [-100]*len(prefix_ids)+target_ids}
