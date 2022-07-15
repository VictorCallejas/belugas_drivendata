import torch

from .data import Eval_BellugaDataset, Train_BellugaDataset, PreTrain_BellugaDataset, METADATA, TRAIN, VAL


PRETRAIN_BS = 6
TRAIN_BS = 128
INFER_BS = TRAIN_BS

NUM_WORKERS = 0


#pretrain_dataset = PreTrain_BellugaDataset(METADATA)
train_train_dataset = Train_BellugaDataset(TRAIN)
#train_eval_dataset = Eval_BellugaDataset(TRAIN)
#valid_eval_dataset = Eval_BellugaDataset(VAL)

'''
pretrain_dataloader = torch.utils.data.DataLoader(
                        pretrain_dataset, 
                        batch_size=PRETRAIN_BS,
                        shuffle=True, 
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )
'''
train_train_dataloader = torch.utils.data.DataLoader(
                        train_train_dataset, 
                        batch_size=TRAIN_BS,
                        shuffle=True, 
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )
'''
train_eval_dataloader = torch.utils.data.DataLoader(
                        train_eval_dataset, 
                        batch_size=INFER_BS,
                        shuffle=True, 
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )

valid_eval_dataloader = torch.utils.data.DataLoader(
                        valid_eval_dataset, 
                        batch_size=INFER_BS,
                        shuffle=False, 
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )   
'''
   