````markdown
# mlops_course_project

README file for Group 82.

1. The dataset we will work with is: Jia555/cis519_news_urls. The short hash for the data we will use is: "6dca88d".

2. The model we would like to work with is microsoft/deberta-v3-base. We will fine tune it for our classification task.

3. The overall goal of the project is to predict the news source by the keywords in the news article URL.

The task it to do binary classification (Fox News vs NBC News). The input is the URL string, which we will engineer so that the news outlet titles are not found in them, and so that the text is solely the key words from the URL.

We intend to use the full dataset to run on. It contains 40858 URLs, of which 19787 is from Fox News and 21721 is from NBC News. The data only contains a single column, which is the full URL from the original news article. The data is text. The output we seek is binary.

We will use logistic regression as our baseline model (combined with TF-IDF).

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
