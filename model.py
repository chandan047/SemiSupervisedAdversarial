import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import (PretrainedConfig, PreTrainedModel,
                          BertConfig, BertForSequenceClassification, BertTokenizer, BertModel,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer, XLMModel,
                          XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, XLNetModel,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertModel,
                          AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel)


MODEL_CLASSES = {
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, 'albert-base-v2'),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer, 'bert-base-uncased'),
    'distilBert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, 'distilbert-base-uncased'),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, 'roberta-base'),
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, 'xlnet-base-cased')
#     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer, 'xlm-mlm-en-2048'),
}


    
def SequenceClassification_fn(model_name, model_path=None):
    encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]

    if model_path is None:
        if model_name == 'xlnet':
            encoder = encoder_class.from_pretrained(options_name, mem_len=1024, num_labels=6)
        else:
            encoder = encoder_class.from_pretrained(options_name, num_labels=6)
    else:
        encoder = encoder_class.from_pretrained(model_path)

    return encoder


class SequenceClassification(PreTrainedModel):
    """model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """
    def __init__(self, model_name, num_labels=6, model_path=None):
        encoder_config, encoder_class, _, options_name = MODEL_CLASSES[model_name]
        config = encoder_config.from_pretrained(options_name) if model_path is None \
                    else encoder_config.from_pretrained(model_path)
        config.num_labels = num_labels
        config.use_return_dict = True
        
        super(SequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.model = encoder_class(config)

    def forward(self, input_ids, input_ids_adv, 
                    token_type_ids=None, attention_mask=None, 
                    token_type_ids_adv=None, attention_mask_adv=None,
                    labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        output_adv = self.model(input_ids=input_ids_adv, attention_mask=attention_mask_adv, labels=labels)

        if output.loss is not None:
            adv_loss = torch.norm(output.logits - output_adv.logits, dim)
            loss = output.loss + adv_loss
            return loss, output.logits
        else:
            return output.logits
        
    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True