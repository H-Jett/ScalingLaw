"""校验平台拓扑到实际训练规模的转换，避免多机 token 预算偏小。"""

import sys
import unittest
from unittest.mock import patch

from scripts.train import resolve_distributed


class DistributedEnvTest(unittest.TestCase):
    def test_two_nodes_four_workers(self):
        env = {"RANK": "1", "WORLD_SIZE": "2", "NPROC_PER_NODE": "4", "MASTER_ADDR": "10.0.0.1"}
        config = resolve_distributed(environ=env)
        self.assertEqual((config.node_rank, config.nnodes, config.world_size), (1, 2, 8))
        child = config.launch_env({**env, "CUSTOM_SETTING": "retained"})
        self.assertEqual(child["NNODES"], "2")
        self.assertEqual(child["NODE_RANK"], "1")
        self.assertEqual(child["NPROC_PER_NODE"], "4")
        self.assertEqual(child["FORCE_TORCHRUN"], "1")
        self.assertEqual(child["CUSTOM_SETTING"], "retained")
        self.assertNotIn("RANK", child)
        self.assertNotIn("WORLD_SIZE", child)

    def test_global_process_semantics(self):
        env = {
            "RANK": "4",
            "WORLD_SIZE": "8",
            "NPROC_PER_NODE": "4",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29600",
        }
        config = resolve_distributed(world_size_unit="processes", environ=env)
        self.assertEqual(
            (config.node_rank, config.nnodes, config.world_size, config.master_port),
            (1, 2, 8, 29600),
        )

    def test_factory_node_variables(self):
        config = resolve_distributed(
            environ={
                "NODE_RANK": "1",
                "NNODES": "2",
                "NPROC_PER_NODE": "4",
                "MASTER_ADDR": "10.0.0.1",
            }
        )
        self.assertEqual((config.node_rank, config.nnodes, config.world_size), (1, 2, 8))

    def test_local_gpu_detection(self):
        # 只模拟设备探测，不需要测试机有 CUDA。
        with patch.dict(sys.modules, {"torch": unittest.mock.Mock()}):
            sys.modules["torch"].cuda.device_count.return_value = 4
            config = resolve_distributed(environ={})
        self.assertEqual((config.nnodes, config.nproc_per_node, config.world_size), (1, 4, 4))

    def test_legacy_world_size_controls_local_launch(self):
        config = resolve_distributed(world_size_override=4, environ={})
        self.assertEqual((config.nnodes, config.nproc_per_node, config.world_size), (1, 4, 4))

    def test_conflicting_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "全局进程数不一致"):
            resolve_distributed(
                4, environ={"WORLD_SIZE": "2", "NPROC_PER_NODE": "4", "MASTER_ADDR": "10.0.0.1"}
            )

    def test_invalid_topology_is_rejected(self):
        cases = [
            {"WORLD_SIZE": "0", "NPROC_PER_NODE": "4"},
            {"WORLD_SIZE": "2", "RANK": "2", "NPROC_PER_NODE": "4", "MASTER_ADDR": "10.0.0.1"},
            {"NPROC_PER_NODE": "0"},
            {"NPROC_PER_NODE": "four"},
            {"WORLD_SIZE": "2", "NPROC_PER_NODE": "4"},
            {"WORLD_SIZE": "2", "NPROC_PER_NODE": "4", "MASTER_ADDR": "127.0.0.1"},
            {"NPROC_PER_NODE": "4", "MASTER_PORT": "65536"},
            {"NPROC_PER_NODE": "4", "LOCAL_RANK": "0"},
            {
                "RANK": "1",
                "NODE_RANK": "0",
                "WORLD_SIZE": "2",
                "NPROC_PER_NODE": "4",
                "MASTER_ADDR": "10.0.0.1",
            },
        ]
        for env in cases:
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve_distributed(environ=env)

    def test_process_mode_requires_one_launcher_per_node(self):
        for rank, size in [(1, 8), (0, 7)]:
            env = {
                "RANK": str(rank),
                "WORLD_SIZE": str(size),
                "NPROC_PER_NODE": "4",
                "MASTER_ADDR": "10.0.0.1",
            }
            with self.subTest(env=env), self.assertRaises(ValueError):
                resolve_distributed(world_size_unit="processes", environ=env)


if __name__ == "__main__":
    unittest.main()
