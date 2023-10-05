import explainer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import tensor
import torchtext

import transformers

from transformers.pipelines import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from captum.concept import TCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.attr import LayerIntegratedGradients, TokenReferenceBase

class TCAVPipeline(explainer.BaseExplainer):
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        super().__init__(name, pipeline, device)
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device

        self.__tcav = TCAV(pipeline, layers=['convs.2', 'convs.1'])
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
        pass
    ##
                      
    def explain(self, text: str, outfile_path: str):
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
    ##
    
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    ##

    def decode(self, inpt: tensor) -> str:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        rebuilt = ""
        
        for i in range(len(inpt)):
            if i + 1 < len(inpt):
                rebuilt += self.__pipeline.tokenizer.decode(inpt[i]) + " "
            else:
                rebuilt += self.__pipeline.tokenizer.decode(inpt[i])
            ##
        ##

        return rebuilt
    ##
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)
    ##

    # Generate a concept's TOKENS
    def get_tensor_from_filename(self, filename):
        ds = torchtext.data.TabularDataset(path=filename,
                                        fields=[('text', torchtext.data.Field()),
                                                ('label', torchtext.data.Field())],
                                        format='csv')
        const_len = 7
        for concept in ds:
            concept.text = concept.text[:const_len]
            concept.text += ['pad'] * max(0, const_len - len(concept.text))
            text_indices = torch.tensor([self.generate_inputs(t) for t in concept.text], device=self.__device)
            yield text_indices
        ##
    ##

    # Generate CONCEPTS
    def assemble_concept(self, name:str, id:int, concepts_path):
        dataset = CustomIterableDataset(self.get_tensor_from_filename, concepts_path)
        concept_iter = dataset_to_dataloader(dataset, batch_size=1)
        return Concept(id=id, name=name, data_iter=concept_iter)
    ##

    # Print concepts
    def print_concept(self, concept_iter):
        cnt = 0
        max_print = 10
        item = next(concept_iter)
        while cnt < max_print and item is not None:
            print(' '.join([self.__pipeline.tokenizer.decode(item_elem) for item_elem in item[0]]))
            item = next(concept_iter)
            cnt += 1
        ##
    ##
##
