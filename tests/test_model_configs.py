"""验证精简后的尺寸生成逻辑与已有模型配置一致。"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import generate_model_configs as configs


REPO = Path(__file__).resolve().parents[1]


class ModelConfigsTest(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((REPO / "models/templates/pythia-14m/config.json").read_text())
        self.args = SimpleNamespace(pin_template=True, ratio=0, count=7, max=1e9, min=10e6)

    def test_default_ladder_matches_committed_configs(self):
        specs, _ = configs.build_ladder(
            self.args, self.base, self.base["vocab_size"], configs.Design()
        )
        self.assertEqual(len(specs), 7)
        for spec in specs:
            path = REPO / "models/configs" / spec["name"] / "config.json"
            self.assertEqual(
                configs.build_config(self.base, spec, seq_len=None, dtype=None),
                json.loads(path.read_text()),
            )

    def test_first_size_can_be_searched_without_pinning_template(self):
        self.args.pin_template = False
        self.args.count = 2
        self.args.max = 1e8
        specs, _ = configs.build_ladder(
            self.args, self.base, self.base["vocab_size"], configs.Design()
        )
        self.assertEqual(len(specs), 2)
        self.assertGreater(specs[1]["nonemb"], specs[0]["nonemb"])


if __name__ == "__main__":
    unittest.main()
