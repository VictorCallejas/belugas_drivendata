import pandas as pd 

from tqdm import tqdm

from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter('ignore')

import torch

import neptune.new as neptune
run = neptune.init(
    project="victorcallejas/Belluga",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNDRlNTJiNC00OTQwLTQxYjgtYWZiNS02OWQ0MDcwZmU5N2YifQ=="
)

from model.CrossVit import VisionTransformer, crossvit_base_448, crossvit_base_244
from utils.scoring import map_score

from data.dataloaders import train_train_dataloader, valid_eval_dataloader
from data.data import getImages

IMAGES = getImages()

device = torch.device("cuda")
        
def main():
    
    model = crossvit_base_244().to(device)

    fp16 = False
    input_dtype = torch.float16 if fp16 else torch.float32
    scaler = torch.cuda.amp.GradScaler()

    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4)
    
    epochs = 500

    for epoch_i in range(0, epochs):
        
        epoch_loss, epoch_acc = 0, 0
        
        model.train()
        
        for anchor, pos, neg in tqdm(train_train_dataloader):

            optimizer.zero_grad(True)
            
            anchor = anchor.to(device, non_blocking=True, dtype=input_dtype)
            pos = pos.to(device, non_blocking=True, dtype=input_dtype)
            neg = neg.to(device, non_blocking=True, dtype=input_dtype)
            
            query = torch.cat([anchor, anchor], dim=0)
            reference = torch.cat([pos, neg], dim=0)
            labels = torch.cat([torch.ones(pos.shape[0],1), torch.zeros(neg.shape[0],1)], dim=0).to(device)

            with torch.cuda.amp.autocast(fp16):
                logits = model(query=query, reference=reference)
                loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            run['running/loss'].log(loss)
            
            # accuracy
            preds = torch.nn.Sigmoid()(logits).round().detach().cpu().numpy()
            acc = accuracy_score(labels.detach().cpu().numpy(), preds)
            run['running/acc'].log(acc)
            
            epoch_loss += loss
            epoch_acc += acc
            
        run['epoch/train/loss'].log(epoch_loss / len(train_train_dataloader))
        run['epoch/train/acc'].log(epoch_acc / len(train_train_dataloader))
        
        if epoch_i % 10 == 0:
            map = map_score(valid_eval_dataloader, model)
            run['epoch5/valid/map'].log(map)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'../artifacts/net_{epoch_i}.pt')


if __name__ == '__main__':
    main()

