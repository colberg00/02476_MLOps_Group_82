## Docker guide:

Build the training image:

```bash
docker build -t mlops82-train:latest -f dockerfiles/train.dockerfile .

Run preprocessing with this command:

docker run --rm -v "%cd%\data:/app/data" mlops82-train:latest \
  uv run python src/mlops_course_project/data.py run-preprocess


Run training with this command

docker run --rm \
  -v "%cd%\data:/app/data" \
  -v "%cd%\models:/app/models" \
  -v "%cd%\reports:/app/reports" \
  mlops82-train:latest \
  uv run python src/mlops_course_project/train.py
