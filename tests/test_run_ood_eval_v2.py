import sys


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None


class _FakeEnv:
    def __init__(self):
        self._done = False

    def reset(self):
        self._done = False
        return 0, {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after done")
        self._done = True
        return (
            0,
            123.0,
            True,
            False,
            {
                "success_rate": 1.0,
                "transfer_efficiency": 0.42,
                "spill_ratio": 0.11,
            },
        )


def test_evaluate_one_reads_task_metric_keys():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    metrics = evaluate_one(_FakeModel(), _FakeEnv(), n_episodes=3)

    assert metrics["success_rate"] == 1.0
    assert metrics["mean_transfer"] == 0.42
    assert metrics["mean_spill"] == 0.11


def test_parse_args_uses_hotfix_residual_scale_default(monkeypatch):
    from pta.scripts.run_ood_eval_v2 import parse_args

    monkeypatch.setattr(sys, "argv", ["run_ood_eval_v2.py"])
    args = parse_args()

    assert args.residual_scale == 0.05
