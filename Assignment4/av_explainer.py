import torch
from torch import tensor

from transformers.pipelines import TextClassificationPipeline

from captum.attr import LayerIntegratedGradients

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class AttentionVisualizerExplainer():
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    ##

    def forward_func(self, inputs: tensor, position = 0):
        """
            The standard forward function that just returns a prediction based on the position in the inputs.
        """
        # Local variable to store the attention mask to be used in explain -- ih-explainer (from class) just tosses this data out
        self.__attention_mask = torch.ones_like(inputs)

        pred = self.__pipeline.model(inputs, attention_mask=self.__attention_mask)

        return pred[position]
    ##

    def forward_func2(self, inputs:tensor):
        # Local variable to store the attention mask to be used in explain -- ih-explainer (from class) just tosses this data out
        self.__attention_mask = torch.ones_like(inputs)
        pred = self.__pipeline.model(inputs, attention_mask=self.__attention_mask)
        return pred.start_logits, pred.end_logits, pred.attentions
    ##
    
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        pass
    ##
                      
    def explain(self, text: str, outfile_path: str):
        # Generate all the baseline information from before
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])

        lig = LayerIntegratedGradients(self.forward_func2, getattr(self.__pipeline.model, 'deberta').embeddings)
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
