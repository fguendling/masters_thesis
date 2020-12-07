#transformers==3.0.2  in requirements-base.txt
# from https://towardsdatascience.com/simple-abstractive-text-summarization-with-pretrained-t5-text-to-text-transfer-transformer-10f6d602c426
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, LongformerModel, LongformerTokenizer

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device('cpu')

def apply_t5_model(text):
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=30,
                                        max_length=100,
                                        early_stopping=True)
    
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)   
    return output

print(apply_t5_model("On the brink of securing his seventh Formula 1 world title, Lewis Hamilton says there are 'no guarantees' that he will continue racing after the 2020 season. The 35-year-old's contract expires at the end of season, and although he has previously said that he'd like to sign an extension, Hamilton is not ruling out the possibility of quitting racing as a whole. Mercedes team principal, Toto Wolff, announced that he plans to step down from his current role after this season, while he will remain with the team. And when asked if Wolff's decision would affect his future, Hamilton said he has considered stepping away. 'I don't know if I'm going to be here next year, so it's not really a concern for me at the moment,' he said after winning Sunday's Emilia Romagna Grand Prix."))

