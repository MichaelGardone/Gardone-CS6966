import torch
from torch import tensor

from transformers.pipelines import TextClassificationPipeline

from captum.attr import LayerIntegratedGradients

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class AttentionVisualizerExplainer():
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device

        if torch.__version__ >= '1.7.0':
            self.__norm_fn = torch.linalg.norm
        else:
            self.__norm_fn = torch.norm
        ##
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

    def forward_func2(self, inputs:tensor, token_type_ids=None, position_ids=None):
        # Local variable to store the attention mask to be used in explain -- ih-explainer (from class) just tosses this data out
        self.__attention_mask = torch.ones_like(inputs)
        output = self.__pipeline.model(inputs, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=self.__attention_mask)
        # print(output)
        return output.logits, output.attentions
    ##
    
    def _visualize_t2t_scores(self, scores_mat, all_tokens, layer, x_label_name='Head', output_dir="out"):
        fig = plt.figure(figsize=(20, 20))
        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx+1)
            # append the attention weights
            im = ax.imshow(scores, cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(all_tokens)))

            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(all_tokens, fontdict=fontdict)
            ax.set_xlabel('{} {}'.format(x_label_name, idx+1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        ##
        plt.tight_layout()

        plt.savefig(output_dir + f"token2token_layer{layer}")
    ##

    def _visualize_token2head_scores(self, scores_mat, all_tokens, output_dir="out"):
        fig = plt.figure(figsize=(30, 50))

        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(6, 2, idx+1)
            # append the attention weights
            im = ax.matshow(scores_np, cmap='viridis')

            fontdict = {'fontsize': 20}

            ax.set_xticks(range(len(all_tokens)))
            ax.set_yticks(range(len(scores)))

            ax.set_xticklabels(all_tokens, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
            ax.set_xlabel('Layer {}'.format(idx+1))

            fig.colorbar(im, fraction=0.046, pad=0.04)
        ##

        plt.tight_layout()
        plt.savefig(output_dir + "token2head")
    ##
    
    def explain(self, text: str, outfile_path: str):
        # Generate all the baseline information from before
        inputs, input_len = self.generate_inputs2(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        token_type_ids, ref_token_type_ids = self.construct_input_ref_token_type_pair(inputs, input_len)
        position_ids, ref_position_ids = self.construct_input_ref_pos_id_pair(inputs)

        indices = inputs[0].detach().tolist()
        all_tokens = self.__pipeline.tokenizer.convert_ids_to_tokens(indices)

        # Take softmax of logits to get the prediction (0 or 1) -- but I don't care about that, I care about the attentions!
        logits, attens = self.forward_func2(inputs, token_type_ids=token_type_ids,position_ids=position_ids)

        indices = inputs[0].detach().tolist()
        all_tokens = self.__pipeline.tokenizer.convert_ids_to_tokens(indices)

        # flatten the tensors into a more easy-to-read form
        all_attens = torch.stack(attens)

        # DeBERTa has 12 layers: [0, 11]
        layer = 2
        self._visualize_t2t_scores(all_attens[layer].squeeze().detach().cpu().numpy(), all_tokens, layer, output_dir=outfile_path)
        return
        
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, 'deberta').embeddings)
        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                  return_convergence_delta = True)
                                #   attribute_to_layer_input = True)
        # We care about inputs in this case, so we want to look at all input attributions rather than output
        
        
        indices = inputs[0].detach().tolist()
        all_tokens = self.__pipeline.tokenizer.convert_ids_to_tokens(indices)

        # print(attributes)
        self._visualize_t2t_scores(attributes[0].squeeze().detach().cpu().numpy(), all_tokens, output_dir=outfile_path)
    ##
    
    def generate_inputs2(self, text: str):
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        encode_text = self.__pipeline.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(encode_text, device = self.__device).unsqueeze(0), len(encode_text)
    ##

    def construct_input_ref_token_type_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=self.__device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.__device)# * -1
        return token_type_ids, ref_token_type_ids
    ##

    def construct_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.__device)
        # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.__device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids

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
