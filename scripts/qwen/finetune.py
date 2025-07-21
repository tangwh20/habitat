import numpy as np

import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

metric = evaluate.load("glue", "mrpc")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    "test-trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none",
    evaluation_strategy="epoch",
    fp16=True,
    deepspeed="/home/tangwenhao/Workspace/habitat/scripts/qwen/ds_config.json",
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])

preds = np.argmax(predictions.predictions, axis=-1)
val_metrics = metric.compute(predictions=preds, references=predictions.label_ids)

breakpoint()
print(tokenized_datasets["train"][0])