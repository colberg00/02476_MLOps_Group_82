# Data workflow (DVC – simplified)

This project uses **DVC** to keep data handling *disciplined* and *explicit*,
without committing large data files to Git.

⚠️ This is **not a full production-ready DVC setup**.
It is a **conceptual and local-only workflow**, meant to enforce good habits
and make it clear **which data is used where**.

---

## What you are expected to do

1. **Download the raw dataset manually**
2. **Place it in the correct folder**
3. **Never commit raw data to Git**
4. **Re-track data with DVC if it changes**
5. **Be explicit when data is used for training or testing**

That’s it.
This setup is about *discipline*, not automation.

---

## Where data lives

- Raw data must be placed in:
    data/raw/

- Example:
    data/raw/news_urls.csv



This folder should contain **only raw, unprocessed data**.

---

## How to get the dataset

Download the dataset once:

```bash
curl -L https://huggingface.co/datasets/Jia555/cis519_news_urls/resolve/main/url_only_data_extra.csv \
   -o data/raw/news_urls.csv

⚠️ The CSV file itself is not tracked by Git.



-- When raw data changes (important) --

If the raw dataset is updated (new rows, cleaned data, new version):

1. Replace the file in data/raw/
2. Re-track it with DVC:

    uv run dvc add data/raw/news_urls.csv

3. Commit only the updated .dvc file:
    - git add data/raw/news_urls.csv.dvc
    - git commit -m "Update raw dataset version"


This makes data changes explicit and traceable.



-- How DVC fits into training and testing --
DVC is not something you run all the time.

Instead, it enforces that:
- training and testing always rely on a defined dataset
- changes to data are intentional, not accidental
- experiments can be reasoned about retrospectively

What this setup does NOT do (yet):
- It does NOT sync data across machines
- It does NOT download data automatically
- It does NOT provide cloud storage

Because of this:
- each machine must download the raw data manually
- this is acceptable for now and intentional


-- Optional: If you f**k up briefly, or just want to try the concept of restoring... --

On the same machine, data can be restored from the local DVC cache:
- uv run dvc checkout

This is only useful if:
- data was deleted by accident
- you want to verify that DVC tracking works
