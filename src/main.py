import os
import warnings

# Set PyTorch MPS fallback BEFORE any PyTorch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging
from typing import List, Optional, Tuple

import pandas as pd
from sb3_contrib.ppo_mask import MaskablePPO

from config import *
from data.calculator import StockDataCalculator
from data.expression import *
from models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from rl.callbacks import CustomCallback
from rl.env.wrapper import AlphaEnv
from rl.policy import Decoder, LSTMSharedNet, TransformerSharedNet
from utils import load_config, reseed_everything, setup_logger
from utils.constants import *

warnings.filterwarnings("ignore")

path_general = Path(load_config("path.yaml")["general"])

# Create logs directory if it doesn't exist
PROJECT_ROOT = Path(__file__).parents[1]  # Go up one level from src to project root
logs_dir = PROJECT_ROOT / "logs"

# Setup root logger with date-based filename
log_file = logs_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger = setup_logger("root", log_file=log_file)

logger = logging.getLogger(__name__)

policy_dict = {"LSTM": LSTMSharedNet, "TRANSFORMER": TransformerSharedNet}


def run_single_experiment(
    data_sources: DataSources,
    alphas: List[Alpha],
    spans: TrainTestSpans,
    groupby: GroupBy,
    group: Group,
    seed: int = SEED,
    pool_capacity: int = POOL_CAPACITY,
    steps: int = STEPS,
):

    reseed_everything(seed)

    logger.info(
        f"""[Main] Starting training process
        Seed: {seed}
        Instruments:
        Pool capacity: {pool_capacity}
        Total Iteration Steps: {steps}"""
    )

    folder_name = f"group-{group.name}_pool-{pool_capacity}_seed-{seed}_policy-{POLICY}_target-{TARGET}"

    save_folder = (
        path_general
        / data_sources.kline.exchange.name
        / data_sources.kline.universe.name
        / "Alphas"
        / data_sources.kline.freq
        / folder_name
    )

    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)

    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    close_ = Feature(Features.CLOSE)

    # ---------------------------------------
    # set the train / big / middle / small / total datasets
    # train dataset for training
    # big / middle / small datasets for ic printing
    # total dataset for computing the full history alpha and save

    data_train = StockData(
        start_time=spans.train_start,
        end_time=spans.train_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=group,
    )

    data_big = StockData(
        start_time=spans.train_start,
        end_time=spans.train_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.BIG,
    )

    data_middle = StockData(
        start_time=spans.train_start,
        end_time=spans.train_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.MIDDLE,
    )

    data_small = StockData(
        start_time=spans.train_start,
        end_time=spans.train_end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.SMALL,
    )

    data_total = StockData(
        start_time=datetime.datetime(2017, 8, 17, 8, 0, 0),
        end_time=datetime.date.today(),
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.ALL,
    )

    target = Dealy(close_, -TARGET) / close_ - 1

    calculator_train = StockDataCalculator(data_train, target)
    calculator_big = StockDataCalculator(data_big, target)
    calculator_middle = StockDataCalculator(data_middle, target)
    calculator_small = StockDataCalculator(data_small, target)
    calculator_total = StockDataCalculator(data_total, target)

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculator_train,
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device,
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    pool = build_pool([])

    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    checkpoint_callback = CustomCallback(
        save_path=save_folder,
        group=group,
        data=data_total,
        calculator_train=calculator_train,
        calculator_big=calculator_big,
        calculator_middle=calculator_middle,
        calculator_small=calculator_small,
        calculator_total=calculator_total,
        policy=POLICY,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=policy_dict[POLICY],
            features_extractor_kwargs=dict(
                n_layers=N_LAYERS,
                d_model=D_MODEL,
                dropout=DROPOUT,
                device=device,
            ),
        ),
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        batch_size=BATCH_SIZE,
        device=device,
        verbose=1,
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=folder_name,
    )


def main(
    data_sources: DataSources,
    alphas: List[Alpha],
    spans: TrainTestSpans,
    groupby: GroupBy,
    group: Group,
    pool_capacity: int = POOL_CAPACITY,
    steps: Optional[int] = STEPS,
):
    """
    :param pool_capacity: Maximum size of the alpha pool
    :param steps: Total iteration steps
    """

    run_single_experiment(
        data_sources=data_sources,
        alphas=alphas,
        spans=spans,
        groupby=groupby,
        group=group,
        pool_capacity=pool_capacity,
        steps=steps,
    )


if __name__ == "__main__":

    # ------------------------------------
    # state the data sources
    data_sources = DataSources(
        kline=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
        group=DataSource(exchange=Exchange.Binance, universe=Universe.spot, freq="1d"),
    )

    # ------------------------------------
    # state all the alpha categories we want to compute
    alphas = [
        Alpha(
            category=Category.liquidity_1d,
            alpha="liq_nonliquidity_10",
        ),
        Alpha(
            category=Category.pv_1d,
            alpha="corr_retvd_30",
        ),
    ]

    # ------------------------------------
    # state the group
    groupby = GroupBy.amount_quarter_spot_3
    group = Group.MIDDLE

    # ------------------------------------
    # state the train test split
    spans = TrainTestSpans(
        train_start=datetime.datetime(2021, 1, 1, 8, 0, 0),
        train_end=datetime.datetime(2025, 6, 1, 8, 0, 0),
    )

    main(
        data_sources=data_sources,
        alphas=alphas,
        spans=spans,
        groupby=groupby,
        group=group,
    )
