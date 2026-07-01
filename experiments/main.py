import argparse
import os
import glob
import itertools
from datetime import datetime

from training.DataLoader import load_dataset
from training.pipeline import (
  run_data_stage,
  run_tuning_stage,
  run_training_stage,
  run_results_stage,
  run_evaluate_stage,
  aggregate_results,
)
from training.tuning_registry import TUNING_REGISTRY
from training import phase_io

RESULTS_DIR = "results"
VALID_STAGES = ["data", "tuning", "training", "results", "evaluate"]

def get_combinations(models):
  combinations = []
  for r in range(2, len(models) + 1):
    combinations.extend(list(itertools.combinations(models, r)))
  return combinations

def resolve_run_dir(args, dataset_name, model_name, run_timestamp):
  if args.from_timestamp:
    return phase_io.get_run_dir_from_timestamp(
      RESULTS_DIR, dataset_name, model_name, args.from_timestamp
    )

  stages = resolve_stages(args.stages)

  if stages[0] == "data":
    return phase_io.get_run_dir(RESULTS_DIR, dataset_name, model_name, run_timestamp)

  return phase_io.get_latest_run_dir(RESULTS_DIR, dataset_name, model_name)

def resolve_stages(stages_arg):
  if "all" in stages_arg:
    return VALID_STAGES

  return [s for s in VALID_STAGES if s in stages_arg]

def run_experiment(model_name, dataset_path, args, run_timestamp, ensemble_models_names=None):
  print(f"\n{'='*60}")
  print(f"Experiment: model={model_name}, dataset={os.path.basename(dataset_path)}")
  if ensemble_models_names:
    print(f"Ensemble components: {ensemble_models_names}")
  print(f"Stages: {args.stages}")
  print(f"{'='*60}")

  dataset_name = os.path.basename(dataset_path).replace(".csv", "")

  save_model_name = model_name
  if model_name == "Ensemble" and ensemble_models_names:
    components_str = "+".join(ensemble_models_names)
    save_model_name = f"Ensemble[{components_str}]"

  run_dir = resolve_run_dir(args, dataset_name, save_model_name, run_timestamp)
  print(f"Run directory: {run_dir}")

  stages = resolve_stages(args.stages)

  if model_name == "Ensemble":
    bw = args.ensemble_backward_window
    fw = args.ensemble_forward_window
    if fw is None:
      raise ValueError("Ensemble requires --ensemble-forward-window to be set for blocked evaluation.")
    
    fixed_params = {"forward_window": fw}
    if bw is not None:
      fixed_params["backward_window"] = bw
    phase_io.save_fixed_params(run_dir, fixed_params)

  for stage in stages:
    if stage == "data":
      run_data_stage(dataset_path, run_dir)

    elif stage == "tuning":
      run_tuning_stage(
        run_dir, model_name, dataset_name,
        n_trials=args.n_trials,
        ensemble_models_names=ensemble_models_names,
        fixed_params=fixed_params if model_name == "Ensemble" else None
      )

    elif stage == "training":
      run_training_stage(
        run_dir, model_name, dataset_name,
        ensemble_models_names=ensemble_models_names,
      )

    elif stage == "results":
      run_results_stage(
        run_dir, model_name, dataset_name,
        ensemble_models_names=ensemble_models_names,
      )

    elif stage == "evaluate":
      run_evaluate_stage(
        run_dir, model_name, dataset_name,
        ensemble_models_names=ensemble_models_names,
      )

  print(f"\nExperiment completed: {save_model_name} / {dataset_name}")


def main():
  parser = argparse.ArgumentParser(
    description="Pipeline - K8S Proactive Autoscaling",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Full pipeline (all stages)
  python main.py --model Prophet --dataset data/cegedim/foo.csv --stages all --n-trials 75

  # Tuning only
  python main.py --model Prophet --dataset data/cegedim/foo.csv --stages tuning --n-trials 100

  # Training + results from latest tuning run
  python main.py --model Prophet --dataset data/cegedim/foo.csv --stages training results

  # Results from a specific prior run
  python main.py --model Prophet --dataset data/cegedim/foo.csv --stages results --from-timestamp 20260531_120000
    """
  )

  parser.add_argument(
    "--model",
    type=str,
    choices=["Prophet", "FFT", "Ensemble", "ExponentialSmoothing", "Transformer", "GRU", "BlockRNNGRU", "all"],
    required=True,
    help="Model to run (e.g. Prophet, FFT, Ensemble) or 'all'",
  )

  parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to dataset CSV or 'all' for all in data/cegedim/",
  )

  parser.add_argument(
    "--ensemble-models",
    nargs="+",
    choices=["Prophet", "FFT", "ExponentialSmoothing", "Transformer", "GRU", "BlockRNNGRU"],
    default=None,
    help="List of models to use in the ensemble (e.g. FFT Prophet)",
  )
  
  parser.add_argument(
    "--exact-ensemble",
    action="store_true",
    help="If set, runs only the exact combination of models provided in --ensemble-models, without generating sub-combinations.",
  )

  parser.add_argument(
    "--n-trials",
    type=int,
    default=75,
    help="Number of Optuna trials for hyperparameter tuning (default: 75, only used by 'tuning' stage)",
  )

  parser.add_argument(
    "--stages",
    nargs="+",
    choices=VALID_STAGES + ["all"],
    default=["all"],
    help=(
      "Pipeline stages to run. One or more of: data, tuning, training, results, all. "
      "Default: all. Example: --stages tuning results"
    ),
  )

  parser.add_argument(
    "--ensemble-backward-window",
    type=int,
    default=None,
    metavar="N",
    help=(
      "Fixed backward_window for Ensemble (required when --model Ensemble). "
      "Not tuned via Optuna. Example: 24 for CPU (freq=1H), 36 for RAM (freq=40T)."
    ),
  )

  parser.add_argument(
    "--ensemble-forward-window",
    type=int,
    default=None,
    metavar="N",
    help="Fixed forward_window for Ensemble (required when --model Ensemble). Typically 1 (one-step-ahead).",
  )

  parser.add_argument(
    "--from-timestamp",
    type=str,
    default=None,
    metavar="YYYYMMDD_HHMMSS",
    help=(
      "Pin downstream stages to a specific prior run directory by timestamp. "
      "If not given, the latest run for this model+dataset is used."
    ),
  )

  args = parser.parse_args()
  run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  if args.dataset.lower() == "all":
    datasets = glob.glob("data/cegedim/*_exp1.csv")
    if not datasets:
      print("No datasets found in data/cegedim/. Exiting.")
      return
  else:
    datasets = [args.dataset]

  if args.model.lower() == "all":
    models_to_run = [m for m in TUNING_REGISTRY.keys() if m != "Ensemble"]
  else:
    models_to_run = [args.model]

  for dataset in datasets:
    for model in models_to_run:
      if model == "Ensemble":
        if not args.ensemble_models:
          print("Error: --ensemble-models must be provided when running Ensemble. Skipping.")
          continue

        if args.exact_ensemble:
          run_experiment(
            "Ensemble", dataset, args, run_timestamp,
            ensemble_models_names=args.ensemble_models,
          )
        else:
          combos = get_combinations(args.ensemble_models)
          for combo in combos:
            run_experiment(
              "Ensemble", dataset, args, run_timestamp,
              ensemble_models_names=list(combo),
            )
      else:
        run_experiment(model, dataset, args, run_timestamp)

  stages = resolve_stages(args.stages)
  if "evaluate" in stages:
    print("\nAggregating all results...")
    aggregate_results(results_dir=RESULTS_DIR)


if __name__ == "__main__":
  main()