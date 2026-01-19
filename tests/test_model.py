from sklearn.pipeline import Pipeline

from mlops_course_project.model import create_baseline_model


def test_create_baseline_model_returns_pipeline():
    model = create_baseline_model()
    assert isinstance(model, Pipeline)


def test_baseline_model_can_fit_and_predict():
    X = [
        "blinken meets qatars prime minister",
        "senators grill ai companies",
        "nyc firefighter dies after blaze",
        "immigration debate heats up",
    ]
    y = [1, 0, 1, 0]  # fox=1, nbc=0

    model = create_baseline_model(
        max_features=100,
        ngram_max=2,
        min_df=1,
        max_iter=100,
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
