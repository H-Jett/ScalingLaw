"""验证 shell 参数入口、训练启动环境和共享配置同步，不启动真实训练。"""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import train
from scripts.train import DistributedEnv

REPO = Path(__file__).resolve().parents[1]


class TrainLaunchTest(unittest.TestCase):
    def test_shell_parameters_and_global_budget(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "RANK",
                "WORLD_SIZE",
                "NPROC_PER_NODE",
                "NNODES",
                "NODE_RANK",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TOKENS",
                "RATIO",
                "WORLD_SIZE_UNIT",
            )
        }
        env.update(
            RANK="1",
            WORLD_SIZE="2",
            NPROC_PER_NODE="4",
            MASTER_ADDR="10.0.0.1",
            RUN_NAME="test-shell-budget",
            MODEL="models/configs/pythia-23m",
            TOKENS="1e8",
            BATCH="2",
            GRAD_ACCUM="3",
            LR="3e-4",
            PYTHON_BIN=sys.executable,
            PYTHONDONTWRITEBYTECODE="1",
        )
        # 从仓库外调用也应正确定位模型和 Python 脚本。
        result = subprocess.run(
            ["bash", str(REPO / "scripts/train.sh"), "--dry-run"],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        budget = json.loads(result.stdout)
        self.assertEqual(budget["n_total"], 22876032)
        self.assertEqual(budget["world_size"], 8)
        self.assertEqual(budget["tokens_per_step"], 2048 * 2 * 3 * 8)
        self.assertEqual(budget["distributed"]["node_rank"], 1)
        self.assertIn("每步 token = 2048 × 2 × 3 × 8", result.stderr)

    def test_training_passes_resolved_topology_and_exit_code(self):
        config = DistributedEnv(1, 2, 4, "10.0.0.1", 29600)
        proc = Mock(stdout=io.StringIO("training output\n"))
        proc.wait.return_value = 7
        # 保留测试日志供检查，不在测试中清理文件。
        run_dir = Path(tempfile.mkdtemp(prefix="scalinglaw-launch-test-"))
        with patch.object(train.subprocess, "Popen", return_value=proc) as popen:
            code = train.run_training(run_dir / "train.yaml", run_dir / "train.log", config)
        self.assertEqual(code, 7)
        command = popen.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "llamafactory.cli", "train"])
        env = popen.call_args.kwargs["env"]
        self.assertEqual((env["NNODES"], env["NODE_RANK"], env["NPROC_PER_NODE"]), ("2", "1", "4"))
        self.assertEqual((env["MASTER_ADDR"], env["MASTER_PORT"]), ("10.0.0.1", "29600"))
        self.assertEqual((run_dir / "train.log").read_text(), "training output\n")

    def test_shared_config_must_match(self):
        run_dir = Path(tempfile.mkdtemp(prefix="scalinglaw-ready-test-"))
        spec = {"arguments": {"lr": 0.0006}, "world_size": 8}
        (run_dir / "train.yaml").write_text("do_train: true\n")
        (run_dir / "launch.json").write_text(json.dumps(spec))
        train.wait_for_config(run_dir, spec, 1)
        with self.assertRaisesRegex(ValueError, "不一致"):
            train.wait_for_config(run_dir, {**spec, "world_size": 4}, 1)


if __name__ == "__main__":
    unittest.main()
