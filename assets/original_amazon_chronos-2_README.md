---
license: apache-2.0
model_id: chronos-2
tags:
  - time series
  - forecasting
  - foundation models
  - pretrained models
  - safetensors
paper:
  - https://arxiv.org/abs/2510.15821
datasets:
  - autogluon/chronos_datasets
  - Salesforce/GiftEvalPretrain
leaderboards:
  - Salesforce/GIFT-Eval
  - autogluon/fev-leaderboard
pipeline_tag: time-series-forecasting
library_name: chronos-forecasting

---

# Chronos-2

**Update Dec 30, 2025:** ☁️ Deploy Chronos-2 on Amazon SageMaker. [New guide](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb) covers real-time GPU and CPU inference, serverless endpoints (run on demand, no idle costs), and batch transform for large-scale forecasting.

**Chronos-2** is a 120M-parameter, encoder-only time series foundation model for zero-shot forecasting.
It supports **univariate**, **multivariate**, and **covariate-informed** tasks within a single architecture.
Inspired by the T5 encoder, Chronos-2 produces multi-step-ahead quantile forecasts and uses a group attention mechanism for efficient in-context learning across related series and covariates.
Trained on a combination of real-world and large-scale synthetic datasets, it achieves **state-of-the-art zero-shot accuracy** among public models on [**fev-bench**](https://huggingface.co/spaces/autogluon/fev-leaderboard), [**GIFT-Eval**](https://huggingface.co/spaces/Salesforce/GIFT-Eval), and [**Chronos Benchmark II**](https://arxiv.org/abs/2403.07815).
Chronos-2 is also **highly efficient**, delivering over 300 time series forecasts per second on a single A10G GPU and supporting both **GPU and CPU inference**.

## Links
- 🚀 [Deploy Chronos-2 on Amazon SageMaker](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb)
- 📄 [Technical report](https://arxiv.org/abs/2510.15821v1)
- 💻 [GitHub](https://github.com/amazon-science/chronos-forecasting)
- 📘 [Example notebook](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/chronos-2-quickstart.ipynb)
- 📰 [Amazon Science Blog](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)


## Overview

| Capability | Chronos-2 | Chronos-Bolt | Chronos |
|------------|-----------|--------------|----------|
| Univariate Forecasting | ✅ | ✅ | ✅ |
| Cross-learning across items | ✅ | ❌ | ❌ |
| Multivariate Forecasting | ✅ | ❌ | ❌ |
| Past-only (real/categorical) covariates | ✅ | ❌ | ❌ |
| Known future (real/categorical) covariates | ✅ | 🧩 | 🧩 |
| Max. Context Length | 8192 | 2048 | 512 |
| Max. Prediction Length | 1024 | 64 | 64 |

🧩 Chronos & Chronos-Bolt do not natively support future covariates, but they can be combined with external covariate regressors (see [AutoGluon tutorial](https://auto.gluon.ai/1.4.0/tutorials/timeseries/forecasting-chronos.html#incorporating-the-covariates)). This only models per-timestep effects, not effects across time. In contrast, Chronos-2 supports all covariate types natively.


## Usage

### Local usage

For experimentation and local inference, you can use the [inference package](https://github.com/amazon-science/chronos-forecasting).

Install the package
```
pip install "chronos-forecasting>=2.0"
```

Make zero-shot predictions using the `pandas` API

```python
import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

# Load historical target values and past values of covariates
context_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet")

# (Optional) Load future values of covariates
test_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet")
future_df = test_df.drop(columns="target")

# Generate predictions with covariates
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=24,  # Number of steps to forecast
    quantile_levels=[0.1, 0.5, 0.9],  # Quantiles for probabilistic forecast
    id_column="id",  # Column identifying different time series
    timestamp_column="timestamp",  # Column with datetime information
    target="target",  # Column(s) with time series values to predict
)
```

### Deploying a Chronos-2 endpoint to SageMaker

For production use, we recommend deploying Chronos-2 endpoints to Amazon SageMaker.

First, update the SageMaker SDK to make sure that all the latest models are available.

```
pip install --upgrade sagemaker
```

Then, deploy an endpoint:

```python
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

role = get_execution_role()

# Deploy Chronos-2
hub = {
    'HF_MODEL_ID': 'amazon/chronos-2',
    'HF_TASK': 'time-series-forecasting',
}

huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,
    transformers_version="4.41",
    pytorch_version="2.1",
    py_version="py310",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
)
```

See the [deployment guide](https://github.com/amazon-science/chronos-forecasting/blob/main/notebooks/deploy-chronos-to-amazon-sagemaker.ipynb) for more details.

## Model Details

- **Model type:** Encoder-only transformer (T5-based)
- **Parameters:** 120M
- **Context length:** 8192 tokens
- **Prediction length:** Up to 1024 steps
- **Training data:** Combination of real-world and synthetic time series datasets
- **License:** Apache 2.0
- **Paper:** [Chronos-2: From univariate to universal forecasting](https://arxiv.org/abs/2510.15821)

## Citation

If you use Chronos-2 in your research, please cite:

```bibtex
@article{ansari2025chronos2,
  title={Chronos-2: From univariate to universal forecasting},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and others},
  journal={arXiv preprint arXiv:2510.15821},
  year={2025}
}
```
