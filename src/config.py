from typing import Type

from data.expression import *

MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [
    # Unary
    Abs,
    WinsorizeStandardize,
    Log,
    # Binary
    Add,
    Sub,
    Mul,
    Div,
    # Rolling
    Dealy,
    Mean,
    Sum,
    Std,
    Var,  # Skew, Kurt,
    Max,
    Min,
    Med,
    Mad,
    Rank,
    Delta,
    Wma,
    Ema,
    Product,
    BollingerBandWidth,
    Rsi,
    MaDeviation,
    Argmax,
    Argmin,
    # # Pair rolling
    Cov,
    Corr,
    Beta,
    ResidualMomentum,
    RightWeightedKurt,
    # If_Else,
]

DELTA_TIMES = [2, 5, 10, 20, 40]

CONSTANTS = [
    -30.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    -0.01,
    0.01,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
]

REWARD_PER_STEP = 0.0
