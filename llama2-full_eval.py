from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

data_files = {
    "train": "/mnt/sdb/benchmark/xiangrui/dfss_new/imdb/train-00000-of-00001.parquet",
    "validation": "/mnt/sdb/benchmark/xiangrui/dfss_new/imdb/test-00000-of-00001.parquet"
}

train_dataset_ = load_dataset('parquet', data_files=data_files, split="train")
eval_dataset_ = load_dataset('parquet', data_files=data_files, split="validation")

train_dataset = train_dataset_.shuffle(seed=42)
eval_dataset = eval_dataset_.shuffle(seed=45)

train_dataset = train_dataset.select([i for i in range(50)])
eval_dataset = eval_dataset.select([i for i in range(15000)])

print(train_dataset)
print(eval_dataset)

t2_num = 0; f2_num = 0

for _ in range(15000):
    if eval_dataset[_]['label'] == 0:
        f2_num = f2_num +1
    else:
        t2_num = t2_num +1

print("label in test_dataset: true_num = ", t2_num, "false_num = ", f2_num)

print("="*20,"loaded dataset","="*20)

# Fine-tuned model name
# new_model = "./trained_model/sql150"
# new_model = "./trained_model/sql500p2/"
new_model = "/data2/share/llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(new_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
print("="*20,"loaded tokenizer","="*20)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

print("="*20,"preprocessed dataset","="*20)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
print("="*20,"loaded accuracy","="*20)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    new_model, num_labels=2, id2label=id2label, label2id=label2id
)
model.config.pad_token_id = model.config.eos_token_id
print("="*20,"loaded model","="*20)

training_args = TrainingArguments(
    output_dir="eval",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    # eval_steps=0.2,
    save_strategy="epoch",
    # save_steps=0.4,
    load_best_model_at_end=True,
    report_to="tensorboard",
    fp16=True,
    # deepspeed="./ds_config_unoffload.json",
    # deepspeed="./ds_config.json",
    # deepspeed="./ds_config_zero3.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

result=trainer.evaluate()
print(result)
# trainer.train()