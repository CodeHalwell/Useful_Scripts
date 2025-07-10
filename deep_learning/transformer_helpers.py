"""Utilities for working with Hugging Face transformers."""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments


def load_model_and_tokenizer(model_name: str):
    """Load a pretrained causal LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 50):
    """Generate text continuation using a causal language model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def text_generation_pipeline(model_name: str):
    """Return a simple text generation pipeline."""
    return pipeline("text-generation", model=model_name)


def fine_tune_text_classification(model_name: str, train_dataset, eval_dataset, epochs: int = 1):
    """Fine-tune a transformer for text classification using Hugging Face Trainer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args = TrainingArguments(output_dir="./tmp_trainer", num_train_epochs=epochs, logging_steps=10)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer)
    trainer.train()
    return trainer
