from pathlib import Path

import pandas as pd

import itertools

import torch
from torchvision import transforms as T
from torchvision.io import read_image

#from CrossVit import crossvit_base_448, crossvit_base_224
from CrossVit3 import crossvit_base_224
#from CrossSwin import cross_swin_tiny_256_8
#from CrossCaiT import vit_concat
#from CrossLeViT import crossvit_base_224

from loguru import logger
import time

START = time.time()

input_dtype = torch.float32

ROOT_DIRECTORY = Path("/code_execution")
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
OUTPUT_FILE = ROOT_DIRECTORY / "submission" / "submission.csv"

METADATA = pd.read_csv(DATA_DIRECTORY / 'metadata.csv')

IMG_SIZE = 224 #224 #448
NORM_TRANSFORMS = torch.nn.Sequential(
    T.Resize([IMG_SIZE, IMG_SIZE]),
    T.ConvertImageDtype(input_dtype),
    T.Normalize(mean = (0.4234, 0.4272, 0.4641),
                std  = (0.2037, 0.2027, 0.2142)),
)

INFER_BS = 300
NUM_WORKERS = 3

SCORE_THRESHOLD = 0.5

def getImages(metadata):
    
    IMAGES = {}
    for image_id, path in zip(metadata.image_id, metadata.path):
        IMAGES[image_id] = NORM_TRANSFORMS(read_image('/code_execution/data/'+path))
    
    return IMAGES

IMAGES = getImages(METADATA)

class Eval_BellugaDataset(torch.utils.data.Dataset):
    def __init__(self, queries_df, database_df):
        
        self.query_df = queries_df
        self.database_df = database_df
        
        self.query_to_image = {}
        for query_row in self.query_df.itertuples():
            self.query_to_image[query_row.query_id] = query_row.query_image_id
        
        self.query_reference = list(itertools.product(self.query_df.query_id.tolist(), self.database_df.database_image_id.tolist()))
        for idx, (query_id, reference_id) in enumerate(self.query_reference):
            if query_id.split('-')[1] == reference_id:
                self.query_reference.pop(idx)
            
    def getimage(self, image_id):
        return IMAGES[image_id]

    def __len__(self):
        return len(self.query_reference)
    
    def __getitem__(self, idx):
        
        query_id = self.query_reference[idx][0]
        query_img_id = self.query_to_image[query_id]
        reference_id = self.query_reference[idx][1]
        
        query = self.getimage(query_img_id)
        reference = self.getimage(reference_id)
        
        return query, reference, query_id, reference_id


def main():
    
    logger.info("Starting main script")
    
    scenarios_df = pd.read_csv(DATA_DIRECTORY / "query_scenarios.csv")

    all_predictions_qid, all_predictions_did, all_predictions_scr = [], [], []
    
    device = torch.device("cuda")
    
    model = crossvit_base_224().to(device)
    ckpt = torch.load('net.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    #model = torch.jit.load('net.pt').to(device)
    
    logger.info("Model loaded")
    
    model.eval()
    
    print(model, flush=True)
    '''
    batch_size = 0
    inc_bs = 100
    print(f'Min. Req. Throughput: {(7_000_000/(3*60*60)):.2f}', flush=True)
    while True:
        batch_size += inc_bs
        dummy_input = torch.randn(batch_size, 3,IMG_SIZE,IMG_SIZE, dtype=input_dtype).to(device)
        repetitions = 50
        total_time = 0
        with torch.no_grad():
            for rep in range(0,repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = model(dummy_input, dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                total_time += curr_time
        Throughput = (repetitions*batch_size)/total_time
        print(f'Throughput with batch_size {batch_size}: {Throughput:.2f}', flush=True)
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f"Total: {t}, Free: {f}, Allocated: {a}, Reserved: {r}", flush=True)
    '''
    sigmoid = torch.nn.Sigmoid()
    global_step = 0
    for scenario, scenario_row in enumerate(scenarios_df.itertuples()):
        
        logger.info(f"Scenario: {scenario}")
        logger.info(f"Elapsed: {int(time.time() - START)}")

        queries_df = pd.read_csv(DATA_DIRECTORY / scenario_row.queries_path)
        database_df = pd.read_csv(DATA_DIRECTORY / scenario_row.database_path)
        
        dataset = Eval_BellugaDataset(queries_df, database_df)
        dataloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=INFER_BS,
                        shuffle=False, 
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )
        
        with torch.no_grad():

            for step, (query, reference, query_id, reference_id) in enumerate(dataloader):
            
                query = query.to(device, non_blocking=True, dtype=input_dtype)
                reference = reference.to(device, non_blocking=True, dtype=input_dtype)

                logits = model(query=query, reference=reference)
                logits = sigmoid(logits).cpu().squeeze().tolist()
                    
                all_predictions_qid.extend(query_id)
                all_predictions_did.extend(reference_id)
                all_predictions_scr.extend(logits)
                global_step+=query.shape[0]
                
                if (step % 300) == 0:
                    elapsed = int(time.time() - START)
                    perc_elapsed = elapsed/(60*60*3)
                    perc_step = global_step/6992235
                    logger.info(f"{(180 - (elapsed/60)):.2f} mins disp. - Elapsed: {perc_elapsed:.3f}% - Step: {perc_step:.3f}%")
    
    logger.info(f"Filtering predictions")
    logger.info(f"Elapsed: {int(time.time() - START)}")       
               
    all_predictions = pd.DataFrame({'query_id':all_predictions_qid, 'database_image_id':all_predictions_did, 'score':all_predictions_scr})
    logger.info(f'NAs:{all_predictions.isna().any().any()}')
    predictions = all_predictions#[all_predictions.score > SCORE_THRESHOLD]
    predictions = predictions[predictions.groupby('query_id')['score'].rank(ascending=False) <= 20]
    #predictions = predictions.groupby('query_id').apply(lambda grp: grp.nlargest(20,'score').sort_values(by='score',ascending=False)).reset_index()
    logger.info(f'{predictions.columns}')
    # For queries where there are no preds bigger than the threshold
    #max_preds = all_predictions[all_predictions.groupby('query_id')['score'].rank(ascending=False) <= 1]
    #logger.info(f'{max_preds.columns}')
    #predictions = pd.concat([predictions, max_preds], axis = 0).drop_duplicates().reset_index(drop=True)
    #logger.info(f'{predictions.columns}')
    predictions.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Elapsed: {int(time.time() - START)}")
    
if __name__ == "__main__":
    main()