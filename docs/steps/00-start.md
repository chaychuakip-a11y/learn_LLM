# 00 从能跑的小例子开始

这一阶段只有 Python，不需要显卡、模型或教师接口。所有命令在仓库根目录运行。先按 README 设置 PYTHONPATH，保证 Python 能找到 src/ticket_sft。

## 先理解四个角色

- ASR 把音频识别成 REC；LAB 是人工标注。
- 诊断 LLM 根据差异、dump 和资源证据决定查什么，写结论。
- 教师是内网 DeepSeek/GLM，用来示范相同任务；输出需要审核。
- 学生是训练后的本地模型，使用相同输入和工具完成任务。

## 第一次真正更新参数

```bash
python -m ticket_sft lab train-toy --steps 80
```

看 `src/ticket_sft/lab.py:train_bigram`。它把“打电话给张三”等短句拆成字符对，为每个前一个字符维护一行“下一个字符”的分数。softmax 把分数变成概率；交叉熵衡量目标字符的概率；梯度 `p-one_hot` 改变参数。终端每10步打印一次真实loss。

这是简单 bigram，不是 Transformer。它帮助你看到“数据→分数→loss→梯度→更新”确实发生了。正式项目用预训练LLM，以免从几百条问题单重新学习语言。

## 再跑完整诊断链

```bash
python -m ticket_sft.intranet demo --out output/first-run
python -m ticket_sft.intranet logs --run output/first-run
```

输入是 `data/example` 的4条合成案例。demo 用已有答案代替教师回复，但脚本执行、规则检查、审核格式和数据构建都实际运行。查看：

```text
output/first-run/candidates/de_nav_002/
  case.snapshot.json    模型看到的案例
  model_1.txt           模拟教师给出的脚本动作
  analysis.py           实际执行的代码
  tool_result.json      ok/exit_code/stdout/stderr
  stdout.json           脚本打印的证据
  model_2.txt           模拟教师读证据后的结论动作
  conclusion.md         可读结论
  transcript.json       完整对话
  result.json           状态、耗时和产物哈希
  events.jsonl          每一步事件
```

目录 `reviewed` 是模拟审核，`data/gemma` 是两条决策一条案例的训练格式。模拟产物带标记，正式训练入口会拒收。

## 动手与验收

先预测：哪一步的输出是模型给的，哪一步是工具执行得到的？再打开 transcript 核对。能解释 `pending_review` 为什么不等于“诊断正确”，即可进入下一步。

Debug：`ModuleNotFoundError` 先检查 PYTHONPATH；`E_OUTPUT_EXISTS` 换新目录；脚本失败先看 stderr.log；不要因为没有显卡跳过这一步。
