"""未来在内网使用：本地CPU合并Gemma4 12B与LoRA，不下载权重。"""
import argparse
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--adapter',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('输出目录已存在，避免覆盖模型版本')
    import torch
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration
    from peft import PeftModel
    base=Gemma4UnifiedForConditionalGeneration.from_pretrained(a.model,dtype=torch.float32,
             device_map={'':'cpu'},local_files_only=True)
    model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True)
    merged=model.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    merged.save_pretrained(a.output,safe_serialization=True,max_shard_size='4GB')
    AutoProcessor.from_pretrained(a.model,local_files_only=True).save_pretrained(a.output)
    print('合并保存完成；还需比较合并前后的固定样本输出。本脚本CPU内存峰值需要实测。')
