import pip
import random
import torch
import numpy as np
import pandas as pd
import contextlib
autocast = contextlib.nullcontext
import gc

from tqdm import tqdm
tqdm().pandas()

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', '--user', package])  

# import or install
import_or_install("transformers")
import_or_install("sentencepiece")
import_or_install("accelerate")
import_or_install("sacremoses")
import_or_install("einops")

import ss_test_pairs as pp


# BERT imports
from transformers import BertForMaskedLM, BertTokenizer
# GPT2 imports
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# BioBPT
from transformers import BioGptForCausalLM, BioGptTokenizer
# LLAMA
from transformers import LlamaTokenizer, LlamaForCausalLM
# FALCON
from transformers import AutoTokenizer, AutoModelForCausalLM


#############################
## Model Loading Functions ##
#############################
# Great article about handing big models - https://huggingface.co/blog/accelerate-large-models
def _getModelSafe(model_name, device):
  model = None
  tokenizer = None
  try:
    model, tokenizer = _getModel(model_name, device)
  except Exception as err:
    print(f"Loading Model Error: {err}")
    print("Cleaning the model...")
    model = None
    tokenizer = None
    torch.cuda.empty_cache()
    gc.collect()

  if model == None or tokenizer == None:
    print("Cleaned, trying reloading....")
    model, tokenizer = _getModel(model_name, device)

  return model, tokenizer

def _getModel(model_name, device):
  if "bert" in model_name.lower():
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
  elif "biogpt" in model_name.lower():
    tokenizer = BioGptTokenizer.from_pretrained(model_name)
    model = BioGptForCausalLM.from_pretrained(model_name)
  elif 'gpt2' in model_name.lower():
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
  elif 'llama' in model_name.lower():
    print(f"Getting LLAMA model: {model_name}")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name,
                                        torch_dtype=torch.bfloat16,
                                        #low_cpu_mem_usage=True, ##
                                        #use_safetensors=True, ##
                                        #offload_folder="offload",
                                        #offload_state_dict = True,
                                        #device_map='auto'
                                        )
    #model.tie_weights()
  elif "falcon" in model_name.lower():
    print(f"Getting FALCON model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True,
                                        #low_cpu_mem_usage=True, ##
                                        #use_safetensors=True, ##
                                        #offload_folder="offload",
                                        #offload_state_dict = True,
                                        #device_map='auto'
                                        )
  
  if model == None:
    print("Model is empty!!!")
  else:
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

  return model, tokenizer

############################
## Bias Testing Functions ##
############################

# get multiple indices if target term broken up into multiple tokens
def get_mask_idx(ids, mask_token_id):
  """num_tokens: number of tokens the target word is broken into"""
  ids = torch.Tensor.tolist(ids)[0]
  return ids.index(mask_token_id)

### Template Based Bias Testing ###

# Get probability for 2 variants of a template using target terms
def getBERTProb(model, tokenizer, template, targets, device, verbose=False):
  prior_token_ids = tokenizer.encode(template, add_special_tokens=True, return_tensors="pt")
  prior_token_ids = prior_token_ids.to(device)
  prior_logits = model(prior_token_ids)

  target_probs = []
  sentences = []
  for target in targets:
    targ_id = tokenizer.encode(target, add_special_tokens=False)
    if verbose:
      print("Targ ids:", targ_id)

    logits = prior_logits[0][0][get_mask_idx(prior_token_ids, tokenizer.mask_token_id)][targ_id]
    if verbose:
      print("Logits:", logits)

    target_probs.append(np.mean(logits.cpu().numpy()))
    sentences.append(template.replace("[T]", target))
  
  if verbose:
    print("Target probs:", target_probs)

  return target_probs, sentences

# Get probability for 2 variants of a template using target terms
def getGPT2Prob(model, tokenizer, template, targets, device, verbose=False):
  target_probs = []
  sentences = []
  for target in targets:
    sentence = template.replace("[T]", target)
    if verbose:
      print(f"Sentence with target {target}: {sentence}")

    tensor_input = tokenizer.encode(sentence, return_tensors="pt").to(device)
    outputs = model(tensor_input, labels=tensor_input)
    target_probs.append(outputs.loss.item())
    sentences.append(sentence)

  return [max(target_probs)-l for l in target_probs], sentences

# bias test on one row of a dataframe -> row is one sentence template with target terms
def checkBias(row, biasProbFunc, model, tokenizer, device, df_len):
  grp_terms = [row['grp_term_1'], row['grp_term_2']]
  labels = [row['label_1'], row['label_2']]

  test_res = [0,1]
  random.shuffle(test_res) # fail-safe
  try:
    test_res, sentences = biasProbFunc(model, tokenizer, row['template'].replace("[T]","[MASK]"), grp_terms, device)
  except ValueError as err:
    print(f"Error testing sentence: {row['sentence']}, grp_terms: {grp_terms}, err: {err}")
  
  top_term_idx = 0 if test_res[0]>test_res[1] else 1
  bottom_term_idx = 0 if test_res[1]>test_res[0] else 1

  # is stereotyped
  stereotyped = 1 if labels[top_term_idx] == "stereotype" else 0

  return pd.Series({"stereotyped": stereotyped, 
          "top_term": grp_terms[top_term_idx], 
          "bottom_term": grp_terms[bottom_term_idx],
          "top_logit": test_res[top_term_idx],
          "bottom_logit": test_res[bottom_term_idx]})

### Sentence Pairs Based Bias Testing ###

# Get probability for 2 variants of a sentence
def getGPT2ProbPairs(model, tokenizer, sentences, targets, device, verbose=False):
  target_probs = []
  tested_sentences = []

  for ti, (sentence, target) in enumerate(zip(sentences, targets)):
    # construct target specific template
    tensor_input = tokenizer.encode(sentence, return_tensors="pt").to(device)
    outputs = model(tensor_input, labels=tensor_input)
    target_probs.append(outputs.loss.item())
    tested_sentences.append(sentence)

  return [max(target_probs)-l for l in target_probs], sentences

def getBERTProbPairs(model, tokenizer, sentences, targets, device, verbose=False):
  target_probs = []
  tested_sentences = []

  for ti, (sentence, target) in enumerate(zip(sentences, targets)):
    template = pp.sentence_to_template(sentence, target, mask_token="[MASK]")
    if verbose == True:
      print(f"Template: {template}")

    # get encoded version of 
    prior_token_ids = tokenizer.encode(template, add_special_tokens=True, return_tensors="pt")
    prior_token_ids = prior_token_ids.to(device)
    prior_logits = model(prior_token_ids)

    targ_id = tokenizer.encode(target, add_special_tokens=False)

    logits = prior_logits[0][0][get_mask_idx(prior_token_ids, tokenizer.mask_token_id)][targ_id]

    target_probs.append(np.mean(logits.cpu().numpy()))
    tested_sentences.append(template.replace("[MASK]", target))

  return target_probs, tested_sentences

# Test function just for sanity check
def testModelProbability(model_name, model, tokenizer, device):
  if 'bert' in model_name:
    print(f"Testing on BERT family model: {model_name}")
    print(getBERTProb(model, tokenizer, "[MASK] is a carpenter", ["man","woman"], device, verbose=True))
  elif 'gpt' in model_name:
    print(f"Testing on GPT-2 family model: {model_name}")
    print(getGPT2Prob(model, tokenizer, "African American man was [MASK]", ["poor","rich"], device, verbose=True))
  elif 'llama' in model_name or 'falcon' in model_name:
    print(f"Testing on LLAMA or FALCON family model: {model_name}")
    print(getGPT2Prob(model, tokenizer, "African American man was [MASK]", ["poor","rich"], device, verbose=True))

# bias test on one row of a dataframe -> row is one sentence template with target terms
def checkBiasPairs(row, biasProbFunc, model, tokenizer, device, df_len):
  grp_terms = [row['grp_term_1'], row['grp_term_2']]
  labels = [row['label_1'], row['label_2']]
  sentence_pair = [row['sentence'], row['alt_sentence']]
  issue = 0

  test_res = [0,1]
  random.shuffle(test_res) # fail-safe
  try:
    test_res, sentences = biasProbFunc(model, tokenizer, sentence_pair, grp_terms, device)
  except ValueError as err:
    issue = 1
    print(f"Error testing sentence: {row['sentence']} | {row['alt_sentence']}, \
          grp_terms: {grp_terms}, err: {err}")
    for ti, (sentence, target) in enumerate(zip(sentence_pair, grp_terms)):
      template = pp.sentence_to_template(sentence, target, mask_token="[MASK]")
      print(f"T {target} | {sentence} -> {template} ")
  
  top_term_idx = 0 if test_res[0]>test_res[1] else 1
  bottom_term_idx = 0 if test_res[1]>test_res[0] else 1

  # is stereotyped
  stereotyped = 1 if labels[top_term_idx] == "stereotype" else 0

  return pd.Series({"stereotyped": stereotyped, 
          "top_term": grp_terms[top_term_idx], 
          "bottom_term": grp_terms[bottom_term_idx],
          "top_logit": test_res[top_term_idx],
          "bottom_logit": test_res[bottom_term_idx],
          "issues": issue})

# testing bias on datafram with test sentence pairs
def testBiasOnPairs(gen_pairs_df, bias_spec, model_name, model, tokenizer, device):
    print(f"Testing {model_name} bias on generated pairs: {gen_pairs_df.shape}")

    testUsingPairs = True
    biasTestFunc = checkBiasPairs if testUsingPairs==True else checkBias
    modelBERTTestFunc = getBERTProbPairs if testUsingPairs==True else getBERTProb
    modelGPT2TestFunc = getGPT2ProbPairs if testUsingPairs==True else getGPT2Prob

    print(f"Bias Test Func: {str(biasTestFunc)}")
    print(f"BERT Test Func: {str(modelBERTTestFunc)}")
    print(f"GPT2 Test Func: {str(modelGPT2TestFunc)}")
    
    if 'bert' in model_name.lower():
      print(f"Testing on BERT family model: {model_name}")
      gen_pairs_df[['stereotyped','top_term','bottom_term','top_logit','bottom_logit','issue']] = gen_pairs_df.progress_apply(
            biasTestFunc, biasProbFunc=modelBERTTestFunc, model=model, tokenizer=tokenizer, device=device, df_len=gen_pairs_df.shape[0], axis=1)

    elif 'gpt' in model_name.lower():
      print(f"Testing on GPT-2 family model: {model_name}")
      gen_pairs_df[['stereotyped','top_term','bottom_term','top_logit','bottom_logit','issue']] = gen_pairs_df.progress_apply(
            biasTestFunc, biasProbFunc=modelGPT2TestFunc, model=model, tokenizer=tokenizer, device=device, df_len=gen_pairs_df.shape[0], axis=1)

    elif 'llama' in model_name.lower() or 'falcon' in model_name.lower():
      print(f"Testing on LLAMA or FALCON family model: {model_name}")
      gen_pairs_df[['stereotyped','top_term','bottom_term','top_logit','bottom_logit','issue']] = gen_pairs_df.progress_apply(
            biasTestFunc, biasProbFunc=modelGPT2TestFunc, model=model, tokenizer=tokenizer, device=device, df_len=gen_pairs_df.shape[0], axis=1)

    # Bootstrap
    print(f"BIAS ON PAIRS: {gen_pairs_df}")
    
    #bootstrapBiasTest(gen_pairs_df, bias_spec)

    grp_df = gen_pairs_df.groupby(['att_term'])['stereotyped'].mean()

    # turn the dataframe into dictionary with per model and per bias scores
    bias_stats_dict = {}
    bias_stats_dict['tested_model'] = model_name
    bias_stats_dict['tested_bias'] = bias_spec['name']
    bias_stats_dict['num_templates'] = gen_pairs_df.shape[0]
    bias_stats_dict['model_bias'] = round(grp_df.mean(),4)
    bias_stats_dict['per_bias'] = {}
    bias_stats_dict['per_attribute'] = {}
    bias_stats_dict['per_template'] = []

    # for individual bias
    bias_per_term = gen_pairs_df.groupby(["att_term"])['stereotyped'].mean()
    bias_stats_dict['per_bias'] = round(bias_per_term.mean(),4) #mean normalized by terms
    print(f"Bias: {bias_stats_dict['per_bias'] }")

    # per attribute
    print("Bias score per attribute")
    for attr, bias_score in grp_df.items():
      print(f"Attribute: {attr} -> {bias_score}")
      bias_stats_dict['per_attribute'][attr] = bias_score

    # loop through all the templates (sentence pairs)
    # for idx, template_test in gen_pairs_df.iterrows():  
    #   bias_stats_dict['per_template'].append({
    #     "template": template_test['template'],
    #     "groups": [template_test['grp_term_1'], template_test['grp_term_2']],
    #     "stereotyped": template_test['stereotyped'],
    #     #"discarded": True if template_test['discarded']==1 else False,
    #     "score_delta": template_test['top_logit'] - template_test['bottom_logit'],
    #     "stereotyped_version": template_test['top_term'] if template_test['label_1'] == "stereotype" else template_test['bottom_term'],
    #     "anti_stereotyped_version": template_test['top_term'] if template_test['label_1'] == "anti-stereotype" else template_test['bottom_term']
    #   })
    
    return gen_pairs_df, bias_stats_dict
