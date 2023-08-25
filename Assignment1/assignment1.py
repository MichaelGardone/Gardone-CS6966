import argparse, os

from datasets import load_dataset, load_metric

import numpy as np

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
progArgs = parser.parse_args()

# The dataset we want to use
task = "imdb"
# The model we want to train
model_checkpoint = "microsoft/deberta-v3-base"
# The batch size -- https://huggingface.co/microsoft/deberta-v3-large showed 8?
batch_size = 2

dataset = load_dataset("imdb")
# print(dataset)
# print(dataset["train"][0]['text'])
metric = load_metric("accuracy", task)
# print(metric)

# Test metric was loaded correctly
# fake_preds = np.random.randint(0, 2, size=(64,))
# fake_labels = np.random.randint(0, 2, size=(64,))
# metric.compute(predictions=fake_preds, references=fake_labels)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint) # , use_fast=True --> doesn't work, should probably confirm this

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)
##

print("Preprocessing data...")
encoded_dataset = dataset.map(preprocess_function, batched=True)
print("Done!")

# Sentiment analysis: only two labels
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    os.path.join(progArgs.output_dir, f"{model_name}-finetuned-{task}"),
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    gradient_accumulation_steps=5,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
##

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate
trainer.evaluate()