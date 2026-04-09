import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datamodule import register_datamodule
from src.datamodule.base import BaseDataModule
from src.message import SupervisionDataModuleMessage


@register_datamodule(name="halloc_text")
class HallocTextDataModule(BaseDataModule):
    def get_datamodule_msg(self) -> SupervisionDataModuleMessage:
        num_total_devices = self.hparams.trainer_msg.num_nodes * self.hparams.trainer_msg.num_devices
        '''
        print(len(self.train_dataset))
        print(num_total_devices)
        print(self.hparams.batch_size_per_gpu)
        print(type(len(self.train_dataset)))
        print(type(num_total_devices))
        print(type(len(self.train_dataset)))
        print(self.hparams.trainer_msg.max_epochs)
        print(type(self.hparams.trainer_msg.max_epochs))
        exit (0)'''

        max_steps = len(self.train_dataset) // (self.hparams.batch_size_per_gpu * num_total_devices) \
            * self.hparams.trainer_msg.max_epochs
        return SupervisionDataModuleMessage(
            classes=self.train_dataset.classes,
            max_steps=max_steps,
        )

    def collate_fn(self, batch):
        has_all_labels = "all_labels" in batch[0]

        #input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        input_ids = torch.from_numpy(np.array([item["input_ids"] for item in batch])).long()
        attention_masks = torch.tensor([item["attention_masks"] for item in batch], dtype=torch.long)
        token_type_ids = torch.tensor([item["token_type_ids"] for item in batch], dtype=torch.long)
        image_paths = [item["image_paths"] for item in batch]

        res = {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "token_type_ids": token_type_ids,
        "image_paths": image_paths,
        }

        if has_all_labels:
            all_labels = torch.tensor([item["all_labels"] for item in batch], dtype=torch.long)
            res["all_labels"] = all_labels
        else:
            label_keys = ["obj_labels", "att_labels", "rel_labels", "sce_labels", "oth_labels"]
            for k in label_keys:
                if k in batch[0]:
                    res[k] = torch.tensor([item[k] for item in batch], dtype=torch.long)

        return res

    '''def collate_fn(self, batch):
        # Helper to check if a key exists in the first item
        has_all_labels = "all_labels" in batch[0]

        # Extract common fields
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_masks = [torch.tensor(item["attention_masks"]) for item in batch]
        token_type_ids = [torch.tensor(item["token_type_ids"]) for item in batch]
        image_paths = [item["image_paths"] for item in batch]

        # Pad common fields
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0) ###this, root cause of 804 length
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

        res = {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "token_type_ids": token_type_ids,
            "image_paths": image_paths,
        }

        if has_all_labels:
            all_labels = [torch.tensor(item["all_labels"]) for item in batch]
            res["all_labels"] = pad_sequence(all_labels, batch_first=True, padding_value=0)
        else:
            # Handle the specific labels since 'all_labels' isn't there
            label_keys = ["obj_labels", "att_labels", "rel_labels", "sce_labels", "oth_labels"]
            for k in label_keys:
                if k in batch[0]:
                    tensors = [torch.tensor(item[k]) for item in batch]
                    res[k] = pad_sequence(tensors, batch_first=True, padding_value=0)
            
            # CRITICAL: If your model expects "all_labels" even in validation, 
            # you must decide what to put there (e.g., a dummy or a copy of obj_labels)
            if "all_labels" in batch[0]: # Extra safety
                 all_labels = [torch.tensor(item["all_labels"]) for item in batch]
                 res["all_labels"] = pad_sequence(all_labels, batch_first=True, padding_value=0)

        return res # Ensure this is ALWAYS returned'''

    '''def collate_fn(self, batch):
        if "all_labels" in batch[0].keys():
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_masks"] for item in batch]
            token_type_ids = [item["token_type_ids"] for item in batch]
            image_paths = [item["image_paths"] for item in batch]
            all_labels = [item["all_labels"] for item in batch]

            input_ids = [torch.tensor(input_id) for input_id in input_ids]
            attention_masks = [torch.tensor(attention_mask) for attention_mask in attention_masks]
            token_type_ids = [torch.tensor(token_type_id) for token_type_id in token_type_ids]
            all_labels = [torch.tensor(labels) for labels in all_labels]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            all_labels = pad_sequence(all_labels, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "token_type_ids": token_type_ids,
                "image_paths": image_paths,
                "all_labels": all_labels,
            }
        else:
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_masks"] for item in batch]
            token_type_ids = [item["token_type_ids"] for item in batch]
            image_paths = [item["image_paths"] for item in batch]
            obj_labels = [item["obj_labels"] for item in batch]
            att_labels = [item["att_labels"] for item in batch]
            rel_labels = [item["rel_labels"] for item in batch]
            sce_labels = [item["sce_labels"] for item in batch]
            oth_labels = [item["oth_labels"] for item in batch]
            all_labels = [item["all_labels"] for item in batch]

            input_ids = [torch.tensor(input_id) for input_id in input_ids]
            attention_masks = [torch.tensor(attention_mask) for attention_mask in attention_masks]
            token_type_ids = [torch.tensor(token_type_id) for token_type_id in token_type_ids]
            obj_labels = [torch.tensor(labels) for labels in obj_labels]
            att_labels = [torch.tensor(labels) for labels in att_labels]
            rel_labels = [torch.tensor(labels) for labels in rel_labels]
            sce_labels = [torch.tensor(labels) for labels in sce_labels]
            oth_labels = [torch.tensor(labels) for labels in oth_labels]
            all_labels = [torch.tensor(labels) for labels in all_labels]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            obj_labels = pad_sequence(obj_labels, batch_first=True, padding_value=0)
            att_labels = pad_sequence(att_labels, batch_first=True, padding_value=0)
            rel_labels = pad_sequence(rel_labels, batch_first=True, padding_value=0)
            sce_labels = pad_sequence(sce_labels, batch_first=True, padding_value=0)
            oth_labels = pad_sequence(oth_labels, batch_first=True, padding_value=0)
            all_labels = pad_sequence(all_labels, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "token_type_ids": token_type_ids,
                "image_paths": image_paths,
                "obj_labels": obj_labels,
                "att_labels": att_labels,
                "rel_labels": rel_labels,
                "sce_labels": sce_labels,
                "oth_labels": oth_labels,
                "all_labels": all_labels
            }'''
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size_per_gpu,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            sampler=sampler,
                            collate_fn=self.collate_fn)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.valid_dataset, shuffle=False)
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn)
