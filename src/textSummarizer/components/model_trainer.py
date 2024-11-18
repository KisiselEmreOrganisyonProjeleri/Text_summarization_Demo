from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from src.textSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Veriyi yükleme
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Eğitim ayarlarını belirleme
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=1000,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=1
        ) 

        # Trainer'ı tanımlama
        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],  # Eğitim veri kümesi
            eval_dataset=dataset_samsum_pt["validation"]  # Değerlendirme veri kümesi
        )

        # Modeli eğitme
        trainer.train()

        # Model ve tokenizer'ı kaydetme
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
