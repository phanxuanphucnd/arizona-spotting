# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn

from fairseq import tasks
from typing import Any, Union
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from arizona.utils.misc_utils import download_url

class Wav2KWS(nn.Module):
    def __init__(
        self,
        num_classes: int=2,
        model_type: str='binary', 
        encoder_hidden_dim: int=768,
        out_channels: int=112,
        pretrained_model: str='wav2vec-base-en'
    ):
        super(Wav2KWS, self).__init__()

        self.num_classes = num_classes
        self.model_type = model_type.lower()
        if model_type == 'binary':
            self.output = 1
        else:
            self.output = self.num_classes

        self.PRETRAINED_MODEL_MAPPING = {
            'wav2vec-base-en': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt',
            'wav2vec-large-en': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt'
        }

        if os.path.isfile(pretrained_model):
            self.pretrained_model_path = os.path.abspath(pretrained_model)            
        elif pretrained_model in self.PRETRAINED_MODEL_MAPPING:
            url = self.PRETRAINED_MODEL_MAPPING[pretrained_model.lower()]
            f_name = pretrained_model + '.pt'
            dest = './.denver/'
            download_url(url, dest, f_name)
            self.pretrained_model_path = os.path.abspath(dest + f_name)
        else:
            raise ValueError(
                f"The `pretrained_model` must be in ['wav2vec-base-en', wav2vec-large-en'] or path to the `Wav2vec` pretrain model. "
            )

        state_dict = torch.load(self.pretrained_model_path)
        
        if state_dict['args'] is None:
            cfg = state_dict['cfg']
        else:
            cfg = convert_namespace_to_omegaconf(state_dict['args'])
        
        task = tasks.setup_task(cfg.task)
        
        self.w2v_encoder = task.build_model(cfg.model)
        self.w2v_encoder.load_state_dict(state_dict['model'])

        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_hidden_dim, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, self.output, 1)
        )

    def forward(self, x):
        output = self.w2v_encoder(**x, features_only=True)
        output = output['x']
        b, t, c = output.shape
        output = output.reshape(b, c, t)
        output = self.decoder(output).squeeze()

        if self.training and self.model_type != 'binary':
            self.softmax = nn.Softmax(dim=-1)
            return self.softmax(output)
        else:
            return output