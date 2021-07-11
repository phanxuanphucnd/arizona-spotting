# -*- coding: utf-8 -*-

import os
import sys
import torch
import shutil
import librosa
import logging
import tempfile
import numpy as np
import torch.nn as nn
import soundfile as sf

from tqdm import tqdm
from typing import Any, List, Union
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from arizona.utils.print_utils import *
from arizona.keyword_spotting.models import Wav2KWS
from arizona.utils.misc_utils import get_from_registry
from arizona.keyword_spotting.datasets import Wav2KWSDataset
from arizona.utils.visualize_utils import plot_confusion_matrix

class Wav2KWSLearner():
    def __init__(
        self,
        model: Wav2KWS=None,
        device: str=None
    ) -> None:
        super(Wav2KWSLearner, self).__init__()

        self.model = model
        self.num_classes = self.model.num_classes
        self.model_type = self.model.model_type

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train(
        self,
        train_dataset: Wav2KWSDataset,
        test_dataset: Wav2KWSDataset,
        batch_size: int=48,
        encoder_learning_rate: float=1e-5,
        decoder_learning_rate: float=5e-4,
        weight_decay: float=1e-5,
        max_steps: Union[int, Any]=None,
        n_epochs: int=100,
        shuffle: bool=True,
        num_workers: int=4,
        view_model: bool=True,
        save_path: str='./models',
        model_name: str='wav2kws_model',
        **kwargs
    ):

        if os.path.exists('log.log'):
            os.remove('log.log')

        logging.basicConfig(filename='log.log', level=logging.INFO)
        
        print_line(text="Dataset Info")
        print(f"Length of Training dataset: {len(train_dataset)}")
        print(f"Length of Test dataset: {len(test_dataset)} \n")

        logging.info(f"classes: {train_dataset.label2idx}")
        logging.info(f"Length of Training dataset: {len(train_dataset)}")
        logging.info(f"Length of Test dataset: {len(test_dataset)} \n")
        
        self._collate_fn = test_dataset._collate_fn
        self.postprocess = test_dataset.postprocess

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, collate_fn=self._collate_fn, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=self._collate_fn, num_workers=num_workers
        )

        self.label2idx = train_dataset.label2idx
        self.idx2label = train_dataset.idx2label
        self.sample_rate = train_dataset.sample_rate

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        criterion = get_build_criterion(model_type=self.model_type)
        optimizer = torch.optim.Adam([
            {'params': self.model.w2v_encoder.parameters(), 'lr': encoder_learning_rate},
            {'params': self.model.decoder.parameters(), 'lr': decoder_learning_rate},
        ], weight_decay=weight_decay)

        criterion.to(self.device)
        self.model.to(self.device)

        # View the architecture of the model
        logging.info("Model Info: ")
        logging.info(self.model)
        if view_model:
            print_line(text="Model Info")
            print(self.model)

        logging.info(f"Using the device: {self.device}")
        print(f"Using the device: {self.device}")

        step = 0
        best_acc = 0
        
        print_line(text="Training the model")
        
        # Check save_path exists
        if not os.path.exists(save_path):
            print(f"\n- Create a folder {save_path}")
            os.makedirs(save_path)

        for epoch in range(n_epochs):
            train_loss, train_acc = self._train(train_dataloader, optimizer, criterion)
            valid_loss, valid_acc = self._validate(test_dataloader, criterion)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n"
                        f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; acc = {valid_acc:.4f} \n"
            )
            logging.info(f"Epoch {epoch + 1}/{n_epochs}: \n"
                         f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f} \n"
                         f"\t- Valid: loss = {valid_loss:.4f}; acc = {valid_acc:.4f} \n")

            if valid_acc > best_acc:
                best_acc = valid_acc
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'label2idx': self.label2idx,
                        'idx2label': self.idx2label,
                        'model_type': self.model_type,
                        'sample_rate': self.sample_rate,
                        # '_collate_fn': self._collate_fn,
                        # 'postprocess': self.postprocess,
                        'loss': valid_loss,
                        'acc': valid_acc
                    },
                    os.path.join(save_path, f"{model_name}.pt")
                )
                print_free_style(f"Save the best model!")
                logging.info(f"Save the best model!")
            else:
                if max_steps:
                    step += 1
                    if step > max_steps:
                        break

    def _train(
        self,
        train_dataloader,
        optimizer,
        criterion,
        model_type='binary'
    ):
        self.model.train()

        correct = 0
        train_loss = []
        total_sample = 0

        for item in tqdm(train_dataloader):
            x, y = item['net_input'], item['target']
            y = y.to(self.device).float()
            for k in x.keys():
                x[k] = x[k].to(self.device)

            optimizer.zero_grad()
            output = self.model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            total_sample += y.size(0)

            if model_type == 'binary':
                pred = (torch.sigmoid(output) >= 0.5).long()
                correct += (pred == y).sum().item()
            else:
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()

        loss = np.mean(train_loss)
        if self.model_type == 'multi-class':
            correct = correct.cpu().detach().numpy()

        acc = correct / total_sample

        return loss, acc
    
    def _validate(
        self,
        valid_dataloader,
        criterion=None,
        model_type='binary'
    ):
        self.model.eval()
        
        correct = 0
        valid_loss = []
        total_sample = 0
        preds = []
        labels = []
        
        with torch.no_grad():
            for item in tqdm(valid_dataloader):
                x, y = item['net_input'], item['target']
                y = y.to(self.device).float()
                for k in x.keys():
                    x[k] = x[k].to(self.device)

                output = self.model(x)
                if criterion:
                    loss = criterion(output, y)
                    valid_loss.append(loss.item())

                total_sample += y.size(0)

                if model_type == 'binary':
                    pred = (torch.sigmoid(output) >= 0.5).long()
                    correct += (pred == y).sum().item()
                else:
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(y.data.view_as(pred)).sum()

                labels.extend(y.view(-1).data.cpu().numpy())
                preds.extend(pred.view(-1).data.cpu().numpy())
        
        loss = np.mean(valid_loss)
        if self.model_type == 'multi-class':
            correct = correct.cpu().detach().numpy()
        acc = correct / total_sample

        return loss, acc

    def load_model(self, model_path):
        """Load the pretained model. """
        # Check the model file exists
        if not os.path.isfile(model_path):
            raise ValueError(f"The model file `{model_path}` is not exists or broken! ")

        checkpoint = torch.load(model_path)
        self.model_type = checkpoint.get('model_type', 'binary')
        self.label2idx = checkpoint.get('label2idx', {})
        self.idx2label = checkpoint.get('idx2label', {})
        self.sample_rate = checkpoint.get('sample_rate', 16000)
        # self._collate_fn = checkpoint.get('_collate_fn', None)
        # self.postprocess = checkpoint.get('postprocess', None)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
    def evaluate(
        self,
        test_dataset: Wav2KWSDataset=None,
        batch_size: int=48,
        num_workers: int=4,
        criterion: Any=None,
        model_type: str='binary',
        view_classification_report: bool=True
    ):
        _collate_fn = test_dataset._collate_fn
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )

        self.model.eval()
        
        correct = 0
        test_loss = []
        total_sample = 0
        preds = []
        labels = []

        print_line(f"Evaluate the model")

        with torch.no_grad():
            for item in tqdm(test_dataloader):
                x, y = item['net_input'], item['target']
                y = y.to(self.device).float()
                for k in x.keys():
                    x[k] = x[k].to(self.device)

                output = self.model(x)
                if criterion:
                    loss = criterion(output, y)
                    test_loss.append(loss.item())

                total_sample += y.size(0)

                if model_type == 'binary':
                    pred = (torch.sigmoid(output) >= 0.5).long()
                    correct += (pred == y).sum().item()
                else:
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(y.data.view_as(pred)).sum()

                labels.extend(y.view(-1).data.cpu().numpy())
                preds.extend(pred.view(-1).data.cpu().numpy())
        
        loss = np.mean(test_loss)
        if self.model_type == 'multi-class':
            correct = correct.cpu().detach().numpy()
        acc = correct / total_sample

        labels = [self.idx2label.get(i) for i in labels]
        preds = [self.idx2label.get(i) for i in preds]
        classes = list(self.label2idx.keys())

        cm = confusion_matrix(y_true=labels, y_pred=preds, labels=classes)
        
        # View classification report
        if view_classification_report:
            report = classification_report(y_true=labels, y_pred=preds, labels=classes)
            print(report)

        # Save confusion matrix image
        try:
            plot_confusion_matrix(cm, target_names=classes, title='confusion_matrix', save_dir='./evaluation')
        except Exception as e:
            print(f"Warning: {e}")

        return loss, acc

    def inference(
        self,
        input: Union[str, List],
    ):
        self.model.eval()

        TEMP_DIR = tempfile.mkdtemp() 
        temp_folder = TEMP_DIR + '/infer/active'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        temp_file = temp_folder + '/test.wav'
        shutil.copyfile(src=input, dst=temp_file)

        test_dataset = Wav2KWSDataset(mode='infer', root=TEMP_DIR)
        _collate_fn = test_dataset._collate_fn
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=4
        )
        pred = None
        with torch.no_grad():
            for item in tqdm(test_dataloader):
                x, _ = item['net_input'], item['target']
                for k in x.keys():
                    x[k] = x[k].to(self.device)
                
                output = self.model(x)
                shutil.rmtree(TEMP_DIR)

                if self.model_type == 'binary':
                    index = (torch.sigmoid(output) >= 0.5).long().tolist()
                    score = torch.sigmoid(output).tolist()
                else:
                    softmax = nn.Softmax(dim=0)
                    output = softmax(output)
                    pred = output.data.max(0, keepdim=True)
                    score, index = pred[0].cpu().numpy()[0], pred[1].cpu().numpy()[0]

        return {
            "idx": index,
            "score": score,
            "name": self.idx2label.get(int(index))
        }


def get_build_criterion(model_type):
    return get_from_registry(model_type, criterion_registry)


criterion_registry = {
    'binary': nn.BCEWithLogitsLoss(),
    'multi-class': nn.CrossEntropyLoss()
}