#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasetss import load_dataset
import transformers

#%%
model_name = "swap-uniba/LLaMAntino-2-7b-hf-ITA"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo

#%%
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#%%
model.eval() # model in evaluation mode (dropout modules are deactivated)

# craft prompt
comment = "Bella risposta, grazie!"
prompt=f'''[INST] {comment} [/INST]'''

# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])
# %%