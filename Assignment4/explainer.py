from torch import tensor
from transformers.pipelines import TextClassificationPipeline

class BaseExplainer():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, name:str, pipeline: TextClassificationPipeline, device: str):
        self.__name = name
        self.__pipeline = pipeline
        self.__device = device
    ##
    
    def forward_func(self, inputs: tensor, position = 0):
        pass
    ##
        
    def visualize(self, inputs: list, attributes: list, outfile_path: str):
        pass
    ##
                      
    def explain(self, text: str, outfile_path: str):
        pass
    ##
    
    def generate_inputs(self, text: str) -> tensor:
        pass
    ##
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        pass
    ##
##
