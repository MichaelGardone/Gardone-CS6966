'''
Code source (with some changes):
https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234
https://gist.githubusercontent.com/theDestI/fe9ea0d89386cf00a12e60dd346f2109/raw/15c992f43ddecb0f0f857cea9f61cd22d59393ab/explain.py
'''

import torch
import torchtext
import pandas as pd
import seaborn as sns

from torch import tensor 
import transformers
from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

#.... Captum imports..................
from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.common import concepts_to_str
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

from torchtext.vocab import Vocab

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset

import matplotlib.pyplot as plt
import numpy as np

import argparse 
import jsonlines
import os 

PRETRAINED_MODEL = 'microsoft/deberta-v3-base'

TEXT = torchtext.data.Field(lower=True, tokenize='spacy')
LABEL = torchtext.data.LabelField(dtype = torch.float)
DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    ##
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
    ##
        
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        #import pdb; pdb.set_trace()
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        cattr = attr.cpu()
        cinputs = inputs.cpu()

        a = pd.Series(cattr.numpy()[0][::-1], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(cinputs.detach().numpy()[0])[::-1])
        
        a.plot.barh(figsize=(10,40))
        plt.savefig(outfile_path)
    ##
                      
    def explain(self, text: str, outfile_path: str):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        tcav = TCAV(self.__pipeline, layers=['convs.2', 'convs.1'])

        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True)

        # Give a path to save
        self.visualize(inputs, attributes, outfile_path)
    ##

    def _extract_scores(interpretations, layer_name, score_type, idx):
        return [interpretations[key][layer_name][score_type][idx].item() for key in interpretations.keys()]
    ##

    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    ##

    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)
    ##
##

# ========= CONCEPT MAKER =========== #
# Generate a concept's TOKENS
def get_tensor_from_filename(filename):
    ds = torchtext.data.TabularDataset(path=filename,
                                       fields=[('text', torchtext.data.Field()),
                                               ('label', torchtext.data.Field())],
                                       format='csv')
    const_len = 7
    for concept in ds:
        concept.text = concept.text[:const_len]
        concept.text += ['pad'] * max(0, const_len - len(concept.text))
        text_indices = torch.tensor([TEXT.vocab.stoi[t] for t in concept.text], device=DEVICE)
        yield text_indices
    ##
##

# Generate CONCEPTS
def assemble_concept(name:str, id:int, concepts_path):
    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
    concept_iter = dataset_to_dataloader(dataset, batch_size=1)
    return Concept(id=id, name=name, data_iter=concept_iter)
##

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
    
    exp_model = ExplainableTransformerPipeline(PRETRAINED_MODEL, clf, DEVICE)

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