"""可执行基础实验。只用标准库，帮助理解真实训练脚本里的几个量。"""
import argparse
import math
import random


def probabilities(logits):
    # 减最大值避免 exp(很大数) 溢出，不改变 softmax 的结果。
    shifted = [math.exp(x - max(logits)) for x in logits]
    total = sum(shifted)
    return [x / total for x in shifted]


def train_bigram(steps=80, rate=1.0, callback=None):
    """真实执行小矩阵的梯度下降。只看前一个字，是教学bigram，不是Transformer。"""
    text = ('打电话给张三。导航到学校。播放音乐。打开空调。' * 4)
    vocab = sorted(set(text))
    ids = {char: i for i, char in enumerate(vocab)}
    pairs = [(ids[a], ids[b]) for a, b in zip(text, text[1:])]
    rng = random.Random(42)
    weights = [[rng.uniform(-0.01, 0.01) for _ in vocab] for _ in vocab]
    losses = []
    for step in range(steps):
        gradient = [[0.0 for _ in vocab] for _ in vocab]
        loss = 0.0
        for context, target in pairs:
            probs = probabilities(weights[context])
            loss -= math.log(max(probs[target], 1e-30))
            # softmax + 交叉熵对logits的导数是 p - one_hot(target)。
            for token in range(len(vocab)):
                gradient[context][token] += (probs[token] - (token == target)) / len(pairs)
        loss /= len(pairs)
        losses.append(loss)
        for context in range(len(vocab)):
            for token in range(len(vocab)):
                weights[context][token] -= rate * gradient[context][token]
        if callback:
            callback(step + 1, loss)
    generated = '打'
    for _ in range(10):
        previous = ids[generated[-1]]
        generated += vocab[max(range(len(vocab)), key=lambda i: weights[previous][i])]
    return losses, generated, len(vocab) ** 2


def main(argv=None):
    p = argparse.ArgumentParser(description='基础实验：loss、mask、LoRA、batch、小矩阵训练')
    p.add_argument('experiment', choices=['loss', 'mask', 'lora', 'batch', 'train-toy'])
    p.add_argument('--steps', type=int, default=80)
    a = p.parse_args(argv)
    if a.experiment == 'loss':
        values = [0.8, 0.5, 0.9]
        for probability in values:
            print(f'正确token概率={probability:.2f}  loss=-log(p)={-math.log(probability):.4f}')
        print(f'平均loss={sum(-math.log(x) for x in values)/len(values):.4f}')
    elif a.experiment == 'mask':
        print('字符级示意，真实Gemma分词用 intranet tokens 查看。')
        for token in ['问题', 'dump', '工具输出', '答案1', '答案2', '结束符']:
            target = token.startswith('答案') or token == '结束符'
            print(f'{token:8s} label={"原token ID" if target else "-100"}  {"参与loss" if target else "只作上下文"}')
    elif a.experiment == 'lora':
        dimension, rank = 4096, 16
        print(f'单个W矩阵：{dimension}×{dimension}={dimension**2:,}个参数（冻结）')
        print(f'A+B：{rank}×{dimension}+{dimension}×{rank}={2*rank*dimension:,}个可训练参数')
        print('W有效 = W + (alpha/rank) × B×A；减少优化器状态，但基座仍需加载。')
    elif a.experiment == 'batch':
        print('每卡1 × 4张卡 × 梯度累积8 = 每次更新32条决策')
        print('480条决策 / 32 = 每epoch约15次更新；一条案例两条决策不等于两个独立案例。')
    else:
        if not 1 <= a.steps <= 10000:
            raise ValueError('steps范围1..10000')
        def show(step, loss):
            if step == 1 or step % 10 == 0 or step == a.steps:
                print(f'step={step:4d} loss={loss:.4f} ' + '#' * max(1, int(loss * 10)), flush=True)
        values, generated, count = train_bigram(a.steps, callback=show)
        print(f'训练了{count}个小矩阵参数，loss {values[0]:.4f} → {values[-1]:.4f}')
        print('贪心生成：' + generated)
        print('这个模型只学习相邻字符；诊断项目使用本地预训练LLM加LoRA。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
