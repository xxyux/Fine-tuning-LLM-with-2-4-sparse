# Fine-tuning-LLM-with-2-4-sparse
Fine-tuning Llama-2-7B for Text classification.

Datasets: imdb , framework: deepspeed.

Refer to [DFSS (ppopp'23)](https://github.com/apuaaChen/DFSS), which dynamically cuts the attention matrix to 2:4 during fine-tuning and inference.

So in this repo, I changed Transformers packge's code, [src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py), which means Sparse Attention Mechanism was applied duing training and inference. 

# Introdution
Model: LLaMA-2-7b. 

Datasets: imdb.
    features: ['text', 'label'].
    
Native llama is not suitable for such tasks.

25000:  'eval_accuracy': 0.49368.

# Fine-tuning
Full parameter fine-tuning, using DeepSpeed-zero-2/3 acceleration, fp16, zero-2, 4 GPUs.

## Run command
1. conda create -n Ft24 python=3.10
2. conda activate Ft24
3. pip install requirements.txt
4. CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 llama2-full.py
5. data set:
train datasets: 500, epoch: 2
eval datasets: 1000
6. Time consumed: 4 GPUs(NVIDIA A6000), batch size=4, about 35 minutes

# Evaluation
Use the checkpoint after fine-tuning to eval the native model and the pruned model (2:4).

## Run command
1. CUDA_VISIBLE_DEVICES=0 python llama2-full_eval.py
2. Time consumption: single card, batch size=1, about 50 minutes.

## Evaluation results:
1. Native llama: 25000: 'eval_accuracy': 0.92032
2. Fine-tuned llama: 25000: 'eval_accuracy': 0.9014
