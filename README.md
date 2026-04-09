# HDC NIDS

A research pipeline for a drift-adaptive hyperdimensional (HDC) network intrusion detection system. This framework implements and tests a memory-augmented continuous learning approach to detect evolving network threats while mitigating catastrophic forgetting.

## 🌟 Key Features

* **Drift-Adaptive Hyperdimensional Computing (HDC):** A specialized NIDS model taking advantage of HDC's robust properties to detect anomalous connections and adapt to concept drift natively.
* **Continual Learning Support:** State-of-the-art dual memory and memory mixing methodologies to retain established patterns while incrementally learning new threat vectors.
* **Stream Evaluation Modes:** Built-in support for prequential evaluation over data streams to properly test models in real-world "online" or "continual" scenarios.
* **Baseline Benchmarking:** Fully integrated baseline architectures including Support Vector Machines (SVM), Multilayer Perceptrons (MLP), and LSTMs for 1:1 performance comparisons.
* **Drop-in Benchmark Datasets:** Integrated loaders and preprocessors tailored for the heavy-hitters of NIDS datasets:
  * CIC-IDS2017
  * UNSW-NB15
* **Configurable Experiment Pipeline:** Drive highly specific scenarios—like stagnation tracking, concept drift learning rates, and memory plasticity—through easy-to-use YAML configurations.

## 🏗️ Architecture

The framework is structured as a modular research harness located in `src/hdc_nids`:

* **`config.py`**: A strictly-typed experiment configuration schema allowing precise control over features, split strategies, latent HD dimensions, and temporal drift adjustments.
* **`models.py / baselines.py`**: Model definitions for the adaptive HDC NIDS along with stateful Scikit-learn/PyTorch implementations of baseline models (LSTM, SVM, MLP).
* **`encoding.py`**: Maps structural network features into hyperdimensional space via efficient spatial/temporal encoding strategies.
* **`runner.py`**: The dynamic experiment loop that provisions prequential streaming environments or static offline training phases.
* **`data.py / preprocessing.py`**: Clean, streamable handlers for CIC-IDS2017 and UNSW-NB15 dataset semantics.
* **`metrics.py` / `plots.py`**: Fine-grained metric collectors tracking F1 score decay, anomaly detection delays, false positive bursts, and system margin reductions.

## 🚀 Installation

Ensure you have Python 3.12+ installed. The environment utilizes `PyTorch` alongside standard scientific staples (`scikit-learn`, `numpy`, `matplotlib`).

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the package locally:
   ```bash
   pip install -e .
   ```

## 🧪 Usage

Experiments are parameterized entirely by configurations in the `configs/` directory.

### Running a Base Experiment
To run an experiment, use the `run_experiment.py` script and pass a configuration YAML:
```bash
python scripts/run_experiment.py configs/cicids_continual_hdc.yaml
```

### Supported Execution Modes
Configured within the YAML (`benchmark_mode` and `stream_mode`), the pipeline supports diverse evaluation methodologies:
* **Offline Processing (`cicids_offline_hdc_tuned.yaml`)**: Traditional training and hold-out testing sequence.
* **Continual Learning (`cicids_continual_lstm.yaml`, `unsw_continual_hdc.yaml`)**: Process stream windows sequentially with online updates, measuring adaptation and retention across long attack streams.

### Executing Direct Comparisons
Dedicated benchmarking scripts run concurrent setups for direct metrics correlation:
```bash
python scripts/benchmark_hdc_vs_lstm.py
python scripts/benchmark_hdc_vs_svm.py
```

## 📂 Project Layout

```text
hdc/
├── configs/       # YAML definitions for offline and continual experiments
├── scripts/       # CLI entrypoints and comparison benchmarks
├── src/hdc_nids/  # Core framework codebase
├── tests/         # Pytest coverage for processing and model steps
├── pyproject.toml # Project packaging configuration
└── README.md 
```
