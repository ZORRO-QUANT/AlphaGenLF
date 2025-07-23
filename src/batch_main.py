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

path_save = Path(load_config("path.yaml")["save_path"])

# Create logs directory if it doesn't exist
PROJECT_ROOT = Path(__file__).parents[1]  # Go up one level from src to project root
logs_dir = PROJECT_ROOT / "logs"

# Setup root logger with date-based filename
log_file = logs_dir / f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logger = setup_logger("root", log_file=log_file)

logger = logging.getLogger(__name__)


# def save_factors(
#     data: StockData,
#     pool: LinearAlphaPool,
#     final_cal: StockDataCalculator,
#     group: Group,
#     batch_size: int = BATCH_SIZE,
#     policy_model: str = "all",
# ):

#     experiment_folder = f"group_{group.name}_batch_{batch_size}_policy_{policy_model}"

#     # todo: modify this path to change the folder of alphas

#     path_factor = (
#         path_save
#         / data.data_sources.kline.exchange.name
#         / data.data_sources.kline.universe.name
#         / "Alphas"
#         / data.data_sources.kline.freq
#         / experiment_folder
#     )

#     if not path_factor.exists():
#         path_factor.mkdir(exist_ok=True, parents=True)

#     dates, symbol = data._dates, data._stock_ids

#     q = data.max_backtrack_days
#     h = -data.max_future_days
#     for expre in pool.state["exprs"]:

#         data = final_cal.evaluate_alpha(expre)

#         df_alpha = pd.DataFrame(data.cpu(), index=dates[q:h], columns=symbol)

#         df_reset = df_alpha.reset_index()
#         df_long = df_reset.melt(
#             id_vars="time", var_name="symbol", value_name=f"{expre}"
#         )
#         df_long.dropna(subset=f"{expre}", inplace=True)
#         df_long.to_csv(f"{path_factor}/{expre}.csv", index=False, encoding="utf-8-sig")

policy_dict = {"LSTM": LSTMSharedNet, "TRANSFORMER": TransformerSharedNet}


def run_single_experiment(
    data_train: StockData,
    data_big: StockData,
    data_middle: StockData,
    data_small: StockData,
    data_total: StockData,
    save_folder,
    folder_name, last_ppo,
    # data_sources: DataSources,
    # alphas: List[Alpha],
    # spans: TrainTestSpans,
    # groupby: GroupBy,
    group: Group,
    seed: int = SEED,
    pool_capacity: int = POOL_CAPACITY,
    steps: int = STEPS,
   
  

):
    device: torch.device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

    close_ = Feature(Features.CLOSE)

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
    if last_ppo:
        model.policy.load_state_dict(last_ppo)

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=folder_name,
    )

    last_ppo = model.policy.state_dict()
    return last_ppo
def get_data(start,end,device,data_sources,alphas,groupby,group):
    data_train = StockData(
        start_time=start,
        end_time=end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=group,
        )
    data_big = StockData(
            start_time=start,
            end_time=end,
            data_sources=data_sources,
            alphas=alphas,
            device=device,
            groupby=groupby,
            group=Group.BIG,
        )

    data_middle = StockData(
        start_time=start,
        end_time=end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.MIDDLE,
        )

    data_small = StockData(
        start_time=start,
        end_time=end,
        data_sources=data_sources,
        alphas=alphas,
        device=device,
        groupby=groupby,
        group=Group.SMALL,
        )
    return data_train,data_big,data_middle,data_small
def one_epoch(
        train_range,
        last_ppo,
        save_folder,
        folder_name,
        pool_capacity,
        steps,
        data_sources,
        alphas,
        device,
        groupby,
        group,
        ):
    data_total = StockData(
            start_time=datetime.datetime(2017, 8, 17, 8, 0, 0),
            end_time=datetime.date.today(),
            data_sources=data_sources,
            alphas=alphas,
            device=device,
            groupby=groupby,
            group=Group.ALL,
        )


    for i in range(1,len(train_range)):
        data_train,data_big,data_middle,data_small = get_data(start=train_range[i-1],
                                                              end=train_range[i],device=device,data_sources=data_sources,alphas=alphas,
                                                              groupby=groupby,group=group,
                                                            )
        current_ppo = run_single_experiment(
            data_train=data_train,
            data_big=data_big,
            data_middle=data_middle,
            data_small=data_small,
            data_total=data_total,
            save_folder=save_folder,
            folder_name=folder_name,
            last_ppo=last_ppo,
            group=group,
            pool_capacity=pool_capacity,
            steps=steps,  
        )
        last_ppo = current_ppo
    return current_ppo

        

def main(
    data_sources: DataSources,
    alphas: List[Alpha],
    spans: TrainTestSpans,
    groupby: GroupBy,
    group: Group,
    pool_capacity: int = POOL_CAPACITY,
    steps: Optional[int] = STEPS,
    seed: int = SEED,
    epoch: int =2,
):
    """
    :param pool_capacity: Maximum size of the alpha pool
    :param steps: Total iteration steps
    """
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
        path_save
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

    # close_ = Feature(Features.CLOSE)

    # ---------------------------------------
    # set the train / big / middle / small / total datasets
    # train dataset for training
    # big / middle / small datasets for ic printing
    # total dataset for computing the full history alpha and save

    train_range = pd.date_range(spans.train_start,spans.train_end,periods=4) 
    
    # for j in range(epoch):
    last_ppo=None
    for i in range(epoch):
        logger.info(f'{i+1}/{epoch}')
        current_ppo=one_epoch(train_range=train_range,last_ppo=last_ppo,save_folder=save_folder,
            folder_name=folder_name,
            pool_capacity=pool_capacity,
            steps=steps,data_sources=data_sources,
            alphas=alphas,device=device,groupby=groupby,group=group)
        last_ppo = current_ppo
        
    
        

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
        train_start=datetime.datetime(2022, 1, 1, 8, 0, 0),
        train_end=datetime.datetime(2025, 6, 1, 8, 0, 0),
    )

    main(
        data_sources=data_sources,
        alphas=alphas,
        spans=spans,
        groupby=groupby,
        group=group,
    )
