from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import OPTForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chai_knowledgde_distillation_enhancement (student_model, teacher_model, tokenizer,dataset_name):
# def chai_knowledgde_distillation_enhancement(student_model, teacher_model, tokenizer):
    epochs=1
    batch_size=16
    temperature=2.0
    alpha=0.5
    print("\n📚 Applying Knowledge Distillation (CHAI-KD)...")

    dataset = load_dataset("glue", "sst2", split="train[:5000]")
    tokenized_dataset = dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length'), batched=True)

    training_args = TrainingArguments(
        output_dir="./chai_kd_model",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        report_to="none"
    )

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ Fix
            labels = inputs.pop("labels")
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits

            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)

            loss = alpha * kd_loss + (1 - alpha) * F.cross_entropy(student_logits, labels)
            return (loss, student_outputs) if return_outputs else loss

    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=128),
        batched=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = KDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator  # ✅ Ensure proper padding
    )



    trainer.train()

    return student_model