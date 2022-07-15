import torch

import pandas as pd

from sklearn.metrics import average_precision_score

from tqdm import tqdm

PREDICTION_LIMIT = 20
QUERY_ID_COL = "query_id"
DATABASE_ID_COL = "database_image_id"
SCORE_COL = "score"

SCORE_THRESHOLD = 0.5

device = torch.device('cuda')

class MeanAveragePrecision:
    @classmethod
    def score(cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int):
        """Calculates mean average precision for a ranking task.
        :param predicted: The predicted values as a dataframe with specified column names
        :param actual: The ground truth values as a dataframe with specified column names
        """
        if not predicted[SCORE_COL].between(0.0, 1.0).all():
            raise ValueError("Scores must be in range [0, 1].")
        if predicted.index.name != QUERY_ID_COL:
            raise ValueError(
                f"First column of submission must be named '{QUERY_ID_COL}', "
                f"got {predicted.index.name}."
            )
        if predicted.columns.to_list() != [DATABASE_ID_COL, SCORE_COL]:
            raise ValueError(
                f"Columns of submission must be named '{[DATABASE_ID_COL, SCORE_COL]}', "
                f"got {predicted.columns.to_list()}."
            )

        unadjusted_aps, predicted_n_pos, actual_n_pos = cls._score_per_query(
            predicted, actual, prediction_limit
        )
        adjusted_aps = unadjusted_aps.multiply(predicted_n_pos).divide(actual_n_pos)
        return adjusted_aps.mean()

    @classmethod
    def _score_per_query(
        cls, predicted: pd.DataFrame, actual: pd.DataFrame, prediction_limit: int
    ):
        """Calculates per-query mean average precision for a ranking task."""
        merged = predicted.merge(
            right=actual.assign(actual=1.0),
            how="left",
            on=[QUERY_ID_COL, DATABASE_ID_COL],
        ).fillna({"actual": 0.0})
        # Per-query raw average precisions based on predictions
        unadjusted_aps = merged.groupby(QUERY_ID_COL).apply(
            lambda df: average_precision_score(df["actual"].values, df[SCORE_COL].values)
            if df["actual"].sum()
            else 0.0
        )
        # Total ground truth positive counts for rescaling
        predicted_n_pos = merged["actual"].groupby(QUERY_ID_COL).sum().astype("int64").rename()
        actual_n_pos = actual.groupby(QUERY_ID_COL).size().clip(upper=prediction_limit)
        return unadjusted_aps, predicted_n_pos, actual_n_pos
    
    
def map_score(dataloader, model):
    
    model.eval()
    
    sub = []
    
    sigmoid = torch.nn.Sigmoid()
    
    with torch.no_grad():             
    
        for query, reference, query_id, reference_id in tqdm(dataloader):
            
            query = query.to(device, non_blocking=True, dtype=torch.float32)
            reference = reference.to(device, non_blocking=True, dtype=torch.float32)

            logits = sigmoid(model(query=query, reference=reference)).cpu().squeeze().tolist()
                
            sub.extend(zip(query_id, reference_id, logits))
            
    sub = pd.DataFrame(sub, columns=['query_id', 'database_image_id', 'score'])
    sub = sub[sub.score > SCORE_THRESHOLD]
    sub = sub.set_index(['database_image_id']).groupby('query_id')['score'].nlargest(20).reset_index()
    sub = sub.set_index('query_id')
    
    mean_avg_prec = MeanAveragePrecision.score(
        predicted=sub, actual=dataloader.dataset.gt, prediction_limit=PREDICTION_LIMIT
    )
    
    print('MaP: ',mean_avg_prec)
    return mean_avg_prec
