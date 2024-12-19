
# In[43]:


from datasets import load_dataset
import numpy as np 
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import BertTokenizerFast 
from transformers import DataCollatorForTokenClassification 
from transformers import AutoModelForTokenClassification 
import evaluate
import random
from datasets import Dataset


# In[44]:


model_name = "intfloat/multilingual-e5-large-instruct"

datasets = load_dataset("kaznerd.py")
label_list = datasets["train"].features["ner_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained(model_name)
example_text = datasets['train'][0]
tokenized_input = tokenizer(example_text["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
word_ids = tokenized_input.word_ids()
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])


# In[45]:


def tokenize_and_align_labels(examples, label_all_tokens=True): 
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None 
        label_ids = []
        for word_idx in word_ids: 
            if word_idx is None: 
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) 
            else: 
                label_ids.append(label[word_idx] if label_all_tokens else -100) 
                 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 


# In[46]:


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_name,num_labels=len(label_list) )

args = TrainingArguments( 
    "main-ner-wlang",
    evaluation_strategy = "epoch", 
    learning_rate=2e-5, 
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=32, 
    num_train_epochs=3, 
    weight_decay=0.001, 
    warmup_steps = 800

) 


# In[47]:


data_collator = DataCollatorForTokenClassification(tokenizer) 
metric = evaluate.load("seqeval") 
example = datasets['train'][0]

labels = [label_list[i] for i in example["ner_tags"]] 
labels


# In[48]:


metric.compute(predictions=[labels], references=[labels]) 


# In[49]:


def compute_metrics(eval_preds): 
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2) 

    predictions = [ 
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(pred_logits, labels) 
    ] 
    
    true_labels = [ 
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(pred_logits, labels) 
   ] 
    results = metric.compute(predictions=predictions, references=true_labels)

    return { 
          "precision": results["overall_precision"], 
          "recall": results["overall_recall"], 
          "f1": results["overall_f1"], 
          "accuracy": results["overall_accuracy"], 
  } 


# In[50]:


# train_dataset = tokenized_datasets["train"].select(range(1000))
trainer = Trainer( 
    model, 
    args, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["validation"], 
    data_collator=data_collator, 
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics,
) 


trainer.train() 


# In[ ]:


model.save_pretrained("main_ner_wlang")
tokenizer.save_pretrained("main_ner_wlang")

import json

id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}


# In[16]:


config = json.load(open("main_ner_wlang/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("main_ner_wlang/config.json","w"))


# In[36]:


from transformers import pipeline
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("main_ner_wlang")

from datasets import Dataset
from transformers import Trainer
import numpy as np
# Step 1: Tokenize the test dataset
tokenized_test_dataset = datasets['test'].map(tokenize_and_align_labels, batched=True)

# Step 2: Define the compute_metrics function (reuse the one you've defined)
def compute_metrics(eval_preds): 
    pred_logits, labels = eval_preds 
    
    pred_logits = np.argmax(pred_logits, axis=2)
    
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    true_labels = [
        [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    
    results = metric.compute(predictions=predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"], 
        "recall": results["overall_recall"], 
        "f1": results["overall_f1"], 
        "accuracy": results["overall_accuracy"], 
    }

# Step 3: Create a new trainer instance for evaluation
trainer = Trainer(
    model=model_fine_tuned,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    eval_dataset = tokenized_datasets["test"]
)

# Step 4: Evaluate the model on the test dataset
results = trainer.evaluate(eval_dataset=tokenized_test_dataset)

# Step 5: Print the results
print("Evaluation on test dataset:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")




# In[16]:


from transformers import pipeline
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("main_ner_wlang")

tokenizer = AutoTokenizer.from_pretrained("main_ner_wlang")

example = """Менің атым Абылай"""

nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer,aggregation_strategy = "none" )

ner_results = nlp(example)

print(ner_results)
token = ""
label_list = []
token_list = []


for result in ner_results:
    if result["word"].startswith("▁"):
        if token:
            token_list.append(token.replace("▁", ""))
        token = result["word"]
        label_list.append(result["entity"])
    else:
        token += result["word"]
token_list.append(token.replace("▁", ""))

for token, label in zip(token_list, label_list):
    print(f"{token}\t{label}")


# In[ ]:




