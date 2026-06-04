from datetime import UTC, datetime

from src.domain.models.market_data import Candle
from tests.market_symbols import ANCHOR


class MockStreamHandler:
    def __init__(self, prices, epoch=1000):
        self.prices = prices
        self._epoch = epoch
        self.candles = {
            ANCHOR: [Candle(ANCHOR, 10.0, 10.0, 10.0, float(p), datetime.now(UTC), self._epoch) for p in prices]
        }

    def get_numpy_series(self, _symbol, _field):
        return self.prices

    def get_last_candle_epoch(self, _symbol):
        return self._epoch


class MockOrchestrator:
    def __init__(self, symbols, prices, *, dl_enabled=True, epoch=1000):
        self.symbols = symbols
        self.config = {
            "data_handler": {"granularity": 300},
            "deep_learning": {
                "enabled": dl_enabled,
                "lookback": 15,
                "training_history_bars": 60,
                "training_epochs": 2,
                "learning_rate": 0.001,
                "validation_bars": 10,
                "min_conviction_execute": 0.50,
                "min_edge_margin": 0.05,
                "min_val_accuracy": 0.0,
                "train_on_new_candle_only": False,
                "model_path_template": "data/dl/{symbol}.pth",
                "deploy_gate": {"enabled": False, "mini_bars": 40},
            },
        }
        self.stream = MockStreamHandler(prices, epoch=epoch)


class MockStreamNoEpochGetter:
    def __init__(self, prices):
        self.prices = prices
        self.candles = {}

    def get_numpy_series(self, _symbol, _field):
        return self.prices
