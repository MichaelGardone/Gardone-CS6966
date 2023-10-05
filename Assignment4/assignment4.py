from tcav_explainer import TCAVPipeline

import torch
import torchtext

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from torchtext.vocab import Vocab

import numpy as np

import argparse 
import jsonlines
import os

PRETRAINED_MODEL = 'microsoft/deberta-v3-base'

# TEXT = torchtext.data.Field(lower=True, tokenize='spacy')
# LABEL = torchtext.data.LabelField(dtype = torch.float)
DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')

# ========= CONCEPT MAKER =========== #
# Print concepts
def print_concept(concept_iter):
    cnt = 0
    max_print = 10
    item = next(concept_iter)
    while cnt < max_print and item is not None:
        print(' '.join([TEXT.vocab.itos[item_elem] for item_elem in item[0]]))
        item = next(concept_iter)
        cnt += 1
    ##
##
# ========= CONCEPT MAKER =========== #

def main(args):
    # Fixing the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL) 
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=args.num_labels, output_attentions=True)
    
    clf = transformers.pipeline("text-classification", 
                                model=model, 
                                tokenizer=tokenizer, 
                                device=DEVICE
                                )
    
    exp_model = TCAVPipeline(PRETRAINED_MODEL, clf, DEVICE)
    
    tf = exp_model.generate_inputs("hello world!")
    print(exp_model.decode(tf))
    
    positive = exp_model.assemble_concept("positive", 0, "data/positive")
    print_concept(iter(positive.data_iter))


    return

    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir, f'example_{idx}'))
            idx+=1
        ##
    ##
##

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a1_analysis_file', type=str, default='data/a1_data.jsonl', help='path to a1 analysis file')
    parser.add_argument('--num_labels', default=2, type=int, help='Task number of labels')
    parser.add_argument('--output_dir', default='out', type=str, help='Directory where model checkpoints will be saved')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
##