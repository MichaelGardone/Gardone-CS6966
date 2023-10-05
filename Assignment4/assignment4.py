from av_explainer import AttentionVisualizerExplainer

import torch

import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import numpy as np

import random
import argparse 
import jsonlines
import os

PRETRAINED_MODEL = 'microsoft/deberta-v3-base'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    exp_model = AttentionVisualizerExplainer(PRETRAINED_MODEL, clf, DEVICE)
    
    idx=0
    with jsonlines.open(args.a1_analysis_file, 'r') as reader:
        for obj in reader:
            exp_model.explain(obj["review"], os.path.join(args.output_dir, f'/example_{idx}/'))
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