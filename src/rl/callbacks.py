import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from data.calculator import StockDataCalculator
from data.stock_data import StockData
from models.linear_alpha_pool import LinearAlphaPool
from rl.env.core import AlphaEnvCore
from utils.constants import Group

logger = logging.getLogger(__name__)


class CustomCallback(BaseCallback):
    def __init__(
        self,
        data: StockData,
        save_path: Path,
        calculator_train: StockDataCalculator,
        calculator_big: StockDataCalculator,
        calculator_middle: StockDataCalculator,
        calculator_small: StockDataCalculator,
        calculator_total: StockDataCalculator,
        group: Group,
        policy: str = "LSTM",
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.data = data
        self.save_path = save_path

        self.group = group
        self.policy = policy

        self.calculator_train = calculator_train
        self.calculator_big = calculator_big
        self.calculator_middle = calculator_middle
        self.calculator_small = calculator_small
        self.calculator_total = calculator_total

        self.last_time = time.time()
        self.fisrt_time = time.time()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:

        assert self.logger is not None
        _, rank_ic_big = self.pool.test_ensemble(self.calculator_big)
        _, rank_ic_middle = self.pool.test_ensemble(self.calculator_middle)
        _, rank_ic_small = self.pool.test_ensemble(self.calculator_small)

        now = time.time()
        epoch_time = now - self.last_time
        total_time = (now - self.fisrt_time) / 60
        self.last_time = now
        self.logger.record("group", self.group.name)
        self.logger.record("policy", self.policy)
        self.logger.record("epoch/time (s)", epoch_time)
        self.logger.record("total/time (min)", total_time)
        self.logger.record("pool/size", self.pool.size)
        self.logger.record(
            "pool/significant",
            (np.abs(self.pool.weights[: self.pool.size]) > 1e-4).sum(),
        )

        self.logger.record("big/rank_ic", rank_ic_big)
        self.logger.record("middle/rank_ic", rank_ic_middle)
        self.logger.record("small/rank_ic", rank_ic_small)

        self.show_pool_state()
        self.save_checkpoint()

    def save_checkpoint(self):

        q = self.data.max_backtrack_days
        h = -self.data.max_future_days

        dates, symbol = self.data._dates, self.data._stock_ids

        for expre in self.pool.state["exprs"]:

            data = self.calculator_total.evaluate_alpha(expre, standardize=False)

            df_alpha = pd.DataFrame(data.cpu(), index=dates[q:h], columns=symbol)

            df_alpha.reset_index(inplace=True)
            df_alpha = df_alpha.melt(
                id_vars="time", var_name="symbol", value_name=f"{expre}"
            )
            df_alpha.dropna(subset=f"{expre}", inplace=True)
            df_alpha.to_csv(
                self.save_path / f"{expre}.csv", index=False, encoding="utf-8-sig"
            )

    def show_pool_state(self):
        state = self.pool.state
        logger.info("---------------------------------------------")
        for i in range(self.pool.size):
            weight = state["weights"][i]
            expr_str = str(state["exprs"][i])
            ic_ret = state["ics_ret"][i]
            print(f"> Alpha #{i}: weight {weight}, expr {expr_str}, IC {ic_ret}")
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print("---------------------------------------------")

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore
