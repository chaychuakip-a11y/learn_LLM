"""Text-only Gemma4 12B LoRA + four-GPU ZeRO-3 starter. GPU run unverified."""
import argparse
import importlib.metadata
import json
import math
import os
import hashlib
from pathlib import Path
from ticket_sft.training_text import encode_example
from ticket_sft.runtime_log import RunLogger

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='本地完整Gemma4 12B IT模型目录')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--deepspeed', default=str(Path(__file__).resolve().parents[2] / 'configs' / 'ds_zero3.json'))
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--epochs', type=float, default=1)
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--save-steps', type=int, default=20)
    p.add_argument('--eval-steps', type=int, default=20)
    return p.parse_args()

def main():
    a = parse()
    if not Path(a.model).is_dir():
        raise ValueError('E_MODEL_PATH: --model 只能指向已经准备好的本地模型目录')
    data_hashes = {n: hashlib.sha256((a.data / n).read_bytes()).hexdigest()
                   for n in ['train.jsonl', 'valid.jsonl', 'split_report.json']}
    preflight_manifest = Path(a.output) / 'training_input.json'
    if a.resume_from_checkpoint:
        if not preflight_manifest.is_file():
            raise ValueError('E_RESUME_MANIFEST: 缺少首次训练的training_input.json，不能确认续训来源')
        previous = json.loads(preflight_manifest.read_text(encoding='utf-8'))
        if previous['data_hashes'] != data_hashes or previous['model_path'] != str(Path(a.model).resolve()):
            raise ValueError('E_RESUME_DATA: 续训数据或基座路径变化，使用新训练版本')
    import torch
    from torch.utils.data import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoProcessor, Gemma4UnifiedForConditionalGeneration,
                              Trainer, TrainerCallback, TrainingArguments)

    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        raise RuntimeError('本配置要求torchrun --nproc_per_node=4；单卡无法装入FP16的12B权重')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank_log = RunLogger(Path(a.output)/'debug'/f'rank_{local_rank}.jsonl', 'training',
                         run_id=Path(a.output).name)
    rank_log.event('training_start', rank=local_rank, arguments=vars(a))
    if local_rank == 0 and not a.resume_from_checkpoint:
        preflight_manifest.write_text(json.dumps({'model_path': str(Path(a.model).resolve()),
                                                 'data_hashes': data_hashes}, indent=2), encoding='utf-8')
    torch.cuda.set_device(local_rank)
    # 先触发实际CUDA计算，捕获不含Volta算子的PyTorch安装。
    smoke = torch.ones((16, 16), device=f'cuda:{local_rank}', dtype=torch.float16)
    assert torch.isfinite(smoke @ smoke).all()
    props = torch.cuda.get_device_properties(local_rank)
    rank_log.event('cuda_smoke_passed', rank=local_rank, gpu=props.name,
                   capability=torch.cuda.get_device_capability(local_rank),
                   vram_bytes=props.total_memory, torch=torch.__version__, cuda=torch.version.cuda)
    print(json.dumps({'rank': local_rank, 'gpu': props.name, 'vram_bytes': props.total_memory,
                      'capability': torch.cuda.get_device_capability(local_rank),
                      'torch': torch.__version__, 'cuda': torch.version.cuda}, ensure_ascii=False))

    # TrainingArguments必须在from_pretrained前创建，使ZeRO-3参与初始化；不使用device_map=auto。
    args = TrainingArguments(
        output_dir=a.output, deepspeed=a.deepspeed, fp16=True, bf16=False,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=a.gradient_accumulation,
        learning_rate=a.learning_rate, num_train_epochs=a.epochs, max_steps=a.max_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch', warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0,
        eval_strategy='steps', eval_steps=a.eval_steps, save_strategy='steps', save_steps=a.save_steps, save_total_limit=2, logging_steps=1, logging_nan_inf_filter=False, report_to='none',
        dataloader_num_workers=0, remove_unused_columns=False, seed=20261028,
        prediction_loss_only=True)
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    class EncodedDataset(Dataset):
        def __init__(self, path):
            self.rows = []
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                try:
                    encoded = encode_example(tokenizer, r['prompt'], r['completion'], a.max_length)
                except Exception as error:
                    rank_log.event('encoding_failed', level='ERROR', path=str(path), line=number,
                                   error_type=type(error).__name__, message=str(error))
                    raise
                self.rows.append(encoded)
            if not self.rows:
                raise ValueError(f'空训练/验证文件: {path}')
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            return self.rows[index]

    train = EncodedDataset(a.data / 'train.jsonl')
    valid = EncodedDataset(a.data / 'valid.jsonl')
    rank_log.event('data_encoded', train_rows=len(train), valid_rows=len(valid),
                   max_train_tokens=max(len(x['input_ids']) for x in train.rows),
                   min_target_tokens=min(sum(t!=-100 for t in x['labels']) for x in train.rows))
    def collate(rows):
        size = max(len(x['input_ids']) for x in rows)
        result = {}
        for key, pad in [('input_ids', tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            result[key] = torch.tensor([x[key]+[pad]*(size-len(x[key])) for x in rows], dtype=torch.long)
        return result

    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.float16, attn_implementation='eager', local_files_only=True)
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    names = []
    suffixes = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in suffixes:
            if not any(x in name.lower() for x in ['audio', 'vision', 'projector', 'multimodal', 'multi_modal']):
                names.append(name)
    if not names:
        raise RuntimeError('未找到语言层LoRA目标；需要根据实际模型模块名适配')
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none', target_modules=names,
        task_type=TaskType.CAUSAL_LM))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    rank_log.event('lora_attached', target_modules=len(names), target_examples=names[:8],
                   allocated_mib=torch.cuda.memory_allocated()/1024**2,
                   reserved_mib=torch.cuda.memory_reserved()/1024**2,
                   peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                   note='ZeRO-3下本地numel可能只是分片大小；目标列表保存在run_manifest。')

    class FiniteGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            rank_log.event('train_metrics', rank=local_rank, step=state.global_step, metrics=logs or {},
                           allocated_mib=torch.cuda.memory_allocated()/1024**2,
                           reserved_mib=torch.cuda.memory_reserved()/1024**2,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
            for key in ['loss', 'eval_loss', 'grad_norm']:
                value = (logs or {}).get(key)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise FloatingPointError(f'非有限{key}: {value}；停止排查FP16稳定性')
            if local_rank == 0:
                print(f"[step {state.global_step}/{state.max_steps}] loss={(logs or {}).get('loss', '-')} "
                      f"grad={(logs or {}).get('grad_norm', '-')} "
                      f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GiB", flush=True)
        def on_save(self, args, state, control, **kwargs):
            rank_log.event('checkpoint_saved', step=state.global_step,
                           path=str(Path(args.output_dir)/f'checkpoint-{state.global_step}'))

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=valid,
                      data_collator=collate, processing_class=processor, callbacks=[FiniteGuard()])
    # 避免把按累计batch归一化的额外kwargs传给不同版本的PEFT/多模态包装层。
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=a.resume_from_checkpoint)
    evaluation = trainer.evaluate()
    # 所有rank都参与ZeRO-3保存；最终产物为LoRA适配器，需要原始模型一起使用。
    trainer.save_model(str(Path(a.output) / 'adapter'))
    rank_log.event('training_completed', step=trainer.state.global_step, evaluation=evaluation)
    if trainer.is_world_process_zero():
        destination = Path(a.output)
        processor.save_pretrained(destination / 'processor')
        manifest = {'model_path': a.model, 'precision': 'fp16', 'lora_targets': names,
                    'arguments': vars(a) | {'data': str(a.data)}, 'evaluation': evaluation, 'global_step': trainer.state.global_step,
                    'versions': {p: importlib.metadata.version(p)
                      for p in ['torch', 'transformers', 'peft', 'accelerate', 'deepspeed']},
                    'data_hashes': {n:hashlib.sha256((a.data/n).read_bytes()).hexdigest()
                       for n in ['train.jsonl','valid.jsonl','split_report.json'] if (a.data/n).is_file()},
                    'note': '验证loss不等于资源定位准确率；仍需独立生成评测。'}
        (destination / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
