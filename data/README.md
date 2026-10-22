# Data

The DA-VINCIS @ IberLEF 2022 data comes from the shared-task organisers and is not redistributed here. Request it through the task and follow their terms of use.

## Expected format

CSV files (`train.csv`, `dev.csv`, `test.csv`) with one tweet per row:

| Column | Content |
|---|---|
| `text` | Tweet text |
| `label` | Task 1: `violent` / `not violent` (also `1`/`0`, `violento` / `no violento`) |
| `category` | Task 2: `accidente`, `homicidio`, `robo`, `secuestro` or `ninguno` (English names also accepted) |

The column names are configurable. For Task 2, point the script at the category column:

```bash
python -m davincis.train --task 2 --label-col category --train data/train.csv --dev data/dev.csv
```

If a tweet can have more than one category, separate them with `;` and add `--multilabel`.

## Sample

[`sample/`](sample/) contains a few **invented** Spanish tweets in this format so the code and tests run without the real data. They are not from DA-VINCIS.
