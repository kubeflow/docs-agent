"""Register the ingest pipelines in Kubeflow Pipelines as recurring runs, or run one now.

    python ingest_schedule.py register --config schedules.json
    python ingest_schedule.py run --config schedules.json --name docs-website
    python ingest_schedule.py run --config schedules.json --name docs-distribution \
        --set collection_name=kubeflow_docs_staging

`register` is what the chart's ingest Job runs after each sync (templates/ingest-schedule-job.yaml):
it compiles the pipelines in this directory, uploads them as a new pipeline version, and
replaces the recurring runs named in the config. Recurring runs that are in the experiment
but no longer in the config are deleted, so the config is the whole schedule.

Config (JSON):

    {
      "experiment": "flo-ingest",
      "schedules": [
        {"name": "docs-website", "pipeline": "docs", "cron": "0 0 2 * * *", "arguments": {}}
      ]
    }

`cron` has six fields (seconds first), as Kubeflow Pipelines expects. `arguments` override
the pipeline's defaults. From a laptop, port-forward first:
kubectl port-forward svc/ml-pipeline 8888:8888 -n kubeflow, then pass --host http://localhost:8888.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PIPELINES_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "http://ml-pipeline.kubeflow.svc.cluster.local:8888"

# key in the config -> (source file, pipeline function)
PIPELINES = {
    "docs": ("kubeflow-pipeline.py", "github_rag_pipeline"),
    "issues": ("issues-pipeline.py", "github_issues_rag_pipeline"),
    "code": ("code-pipeline.py", "code_rag_pipeline"),
}


def load_config(path: str) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    schedules = config.get("schedules") or []
    names = [schedule.get("name") for schedule in schedules]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("every schedule needs a unique name")
    for schedule in schedules:
        if schedule.get("pipeline") not in PIPELINES:
            raise ValueError(f"{schedule['name']}: pipeline must be one of {sorted(PIPELINES)}")
        if len(str(schedule.get("cron", "")).split()) != 6:
            raise ValueError(f"{schedule['name']}: cron needs six fields (seconds first)")
        if not isinstance(schedule.get("arguments", {}), dict):
            raise ValueError(f"{schedule['name']}: arguments must be an object")
    config.setdefault("experiment", "flo-ingest")
    return config


def compile_pipeline(key: str, out_dir: str) -> str:
    """Compile one pipeline to YAML and return the file path."""
    from kfp import compiler

    file_name, func_name = PIPELINES[key]
    for extra in (PIPELINES_DIR, PIPELINES_DIR / "utils"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location(f"flo_ingest_{key}", PIPELINES_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package_path = os.path.join(out_dir, f"{key}.yaml")
    compiler.Compiler().compile(pipeline_func=getattr(module, func_name), package_path=package_path)
    return package_path


def upload_version(client, key: str, package_path: str, version: str):
    """Upload the compiled pipeline as a new version of flo-ingest-<key>."""
    pipeline_name = f"flo-ingest-{key}"
    pipeline_id = client.get_pipeline_id(pipeline_name)
    if pipeline_id is None:
        pipeline = client.upload_pipeline(package_path, pipeline_name=pipeline_name)
        pipeline_id = pipeline.pipeline_id
    uploaded = client.upload_pipeline_version(package_path, version, pipeline_id=pipeline_id)
    return pipeline_id, uploaded.pipeline_version_id


def existing_recurring_runs(client, experiment_id: str) -> list:
    runs, token = [], ""
    while True:
        page = client.list_recurring_runs(page_token=token, page_size=100, experiment_id=experiment_id)
        runs.extend(page.recurring_runs or [])
        token = page.next_page_token or ""
        if not token:
            return runs


def register(client, config: dict, version: str) -> None:
    experiment = client.create_experiment(config["experiment"])
    experiment_id = experiment.experiment_id

    with tempfile.TemporaryDirectory() as out_dir:
        versions = {}
        for key in sorted({schedule["pipeline"] for schedule in config["schedules"]}):
            versions[key] = upload_version(client, key, compile_pipeline(key, out_dir), version)
            print(f"uploaded flo-ingest-{key} version {version}")

    # Create the new recurring runs first, then delete the old ones, so a failure half
    # way leaves the previous schedule in place rather than none.
    old = existing_recurring_runs(client, experiment_id)
    for schedule in config["schedules"]:
        pipeline_id, version_id = versions[schedule["pipeline"]]
        client.create_recurring_run(
            experiment_id=experiment_id,
            job_name=schedule["name"],
            description=f"flo ingest ({version})",
            cron_expression=schedule["cron"],
            max_concurrency=1,
            no_catchup=True,
            params=schedule.get("arguments", {}),
            pipeline_id=pipeline_id,
            version_id=version_id,
            enabled=schedule.get("enabled", True),
            enable_caching=False,
        )
        print(f"scheduled {schedule['name']}: {schedule['cron']}")
    for run in old:
        client.delete_recurring_run(run.recurring_run_id)
        print(f"removed previous recurring run {run.display_name}")


def run_once(client, config: dict, name: str, overrides: dict) -> str:
    schedule = next((s for s in config["schedules"] if s["name"] == name), None)
    if schedule is None:
        raise SystemExit(f"no schedule named {name!r}; have {[s['name'] for s in config['schedules']]}")
    arguments = {**schedule.get("arguments", {}), **overrides}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.TemporaryDirectory() as out_dir:
        result = client.create_run_from_pipeline_package(
            pipeline_file=compile_pipeline(schedule["pipeline"], out_dir),
            arguments=arguments,
            run_name=f"{name}-manual-{stamp}",
            experiment_name=config["experiment"],
            enable_caching=False,
        )
    print(f"started run {result.run_id} ({name}) with {arguments}")
    return result.run_id


def parse_overrides(pairs: list[str]) -> dict:
    """key=value pairs; values are parsed as JSON when possible (true, 300, "text")."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["register", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default=os.environ.get("KFP_HOST", DEFAULT_HOST))
    parser.add_argument("--name", help="schedule to run (run only)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="argument override (run only)")
    args = parser.parse_args(argv)

    import kfp

    config = load_config(args.config)
    client = kfp.Client(host=args.host)
    if args.command == "register":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        register(client, config, f"{os.environ.get('INGEST_VERSION', 'dev')}-{stamp}")
    else:
        if not args.name:
            raise SystemExit("run needs --name")
        run_once(client, config, args.name, parse_overrides(args.set))


if __name__ == "__main__":
    main()
