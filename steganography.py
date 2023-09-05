# imports
import torch
import time
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
from dataset import SteganographyDataset
import config as cfg
from reward import reward_fn


dev = "cuda" if torch.cuda.is_available() else "cpu"
dataset = SteganographyDataset()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# get models
model_name = "EleutherAI/gpt-j-6B"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, 
        peft_config = lora_config,
        load_in_8bit=True
    )

model_ref = create_reference_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# initialize trainer
ppo_config = PPOConfig(
    batch_size=cfg.BATCH_SIZE,
    learning_rate = 1e-6
)

ppo_trainer = PPOTrainer(
        ppo_config, 
        model, 
        model_ref, 
        tokenizer, 
        dataset=dataset)
# encode a query

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 1000,
}

for i in range(cfg.NUM_BATCHES):
    query_txt, query_answers = list(
            zip(*[dataset[0] for _ in range(cfg.BATCH_SIZE)]))
    #query_txt = ["PLACEHOLDER"] * batch_size
    query_encoding = tokenizer.batch_encode_plus(query_txt, return_tensors="pt")
    query_list = list(query_encoding["input_ids"])

    # get model response
    #response_list  = respond_to_batch(model, query_list.to(dev))
    response_list = ppo_trainer.generate(query_list, batch_size=cfg.BATCH_SIZE, **generation_kwargs)
    
    reward = [reward_fn(tokenizer.decode(response), answer)
              for response, answer in zip(response_list, query_answers)]

    # train model for one step with ppo
    train_stats = ppo_trainer.step(query_list, response_list, reward)

ppo_trainer.save_model(f'models/{model_name}-{time.time()}')

