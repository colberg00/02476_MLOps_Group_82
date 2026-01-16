from torch import nn
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def create_baseline_model(
    seed: int = 42,
    max_features: int = 50_000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    min_df: int = 2,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Pipeline:
    """Create TF-IDF + Logistic Regression pipeline for news outlet classification."""
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(ngram_min, ngram_max),
                    min_df=min_df,
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    random_state=seed,
                    C=C,
                    max_iter=max_iter,
                    class_weight="balanced",
                    n_jobs=None,
                ),
            ),
        ]
    )
    return pipeline


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
