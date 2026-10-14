"""用小型 parquet 和模拟训练框架验证 shell 的三个独立步骤。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        # 产物留在 /tmp 便于检查；测试不清理或删除任何文件。
        self.root = Path(tempfile.mkdtemp(prefix="scalinglaw-pipeline-test-"))
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        for i in range(3):
            pq.write_table(
                pa.table({"text": ["example training document"] * 16, "token_count": [16] * 16}),
                self.corpus / f"{i}.parquet",
            )
        package = self.root / "llamafactory"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "cli.py").write_text("""import json, os, sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[-1]).read_text())
data_dir = Path(config["dataset_dir"])
assert (data_dir / "dataset_info.json").is_file(), "data must be prepared before training"
assert (data_dir / "corpus.json").is_file(), "corpus metadata must exist"
assert list((data_dir / "raw").glob("*.parquet")), "no prepared shards"
code = int(os.environ.get("FAKE_TRAIN_EXIT", "0"))
if code:
    sys.exit(code)
out = Path(config["output_dir"])
records = [{"current_steps": 1, "loss": 10.0, "lr": config["learning_rate"]},
           {"current_steps": config["max_steps"], "loss": 6.0, "lr": 0.0}]
(out / "trainer_log.jsonl").write_text("".join(json.dumps(r) + "\\n" for r in records))
""")
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "RANK",
                "WORLD_SIZE",
                "NNODES",
                "NODE_RANK",
                "LOCAL_RANK",
                "NPROC_PER_NODE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TOKENS",
                "RATIO",
                "WORLD_SIZE_UNIT",
                "WARMUP_STEPS",
                "SAVE_STEPS",
                "ALLOW_REUSE",
                "GRAD_CHECKPOINTING",
                "SAVE_FINAL",
            )
        }
        self.env.update(
            RANK="0",
            WORLD_SIZE="1",
            NPROC_PER_NODE="1",
            MASTER_ADDR="127.0.0.1",
            MODEL="models/configs/pythia-14m",
            RUN_NAME="pipeline",
            TOKENS="128",
            SEQ="8",
            BATCH="1",
            GRAD_ACCUM="1",
            WARMUP_STEPS="1",
            CORPUS=str(self.corpus),
            RUN_ROOT=str(self.root / "runs"),
            KEEP_MODEL="1",
            SWANLAB_MODE="off",
            PYTHON_BIN=sys.executable,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(self.root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        )

    def run_pipeline(self, *args):
        return subprocess.run(
            ["bash", str(REPO / "scripts/train.sh"), *args],
            cwd="/tmp",
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_data_training_and_plot_run_in_order(self):
        # CLI 覆盖必须同时影响数据准备与实际训练，不能仅改变最后一个步骤。
        result = self.run_pipeline("--tokens", "160", "--name", "overridden")
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / "runs" / "overridden"
        self.assertTrue((run / "data" / "dataset_info.json").is_file())
        self.assertTrue((run / "data" / "corpus.json").is_file())
        self.assertTrue((run / "train.yaml").is_file())
        self.assertTrue((run / "loss.png").is_file())
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["last_loss"], 6.0)
        self.assertEqual(summary["total_tokens"], 160)
        markers = [result.stderr.index(f"[{i}/3]") for i in (1, 2, 3)]
        self.assertEqual(markers, sorted(markers))

    def test_training_failure_stops_before_plot(self):
        self.env["FAKE_TRAIN_EXIT"] = "7"
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertNotIn("[3/3]", result.stderr)
        self.assertFalse((self.root / "runs" / "pipeline" / "loss.png").exists())

    def test_cli_ratio_replaces_default_token_budget(self):
        result = self.run_pipeline("--ratio", "1e-5")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root / "runs" / "pipeline" / "summary.json"
        self.assertEqual(json.loads(path.read_text())["total_tokens"], 144)

    def test_data_failure_stops_before_training(self):
        self.env["CORPUS"] = str(self.root / "missing-corpus")
        result = self.run_pipeline()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("[2/3]", result.stderr)
        self.assertFalse((self.root / "runs" / "pipeline" / "out").exists())


if __name__ == "__main__":
    unittest.main()
