import datetime
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Literal, Optional, Union


class GroupBy(IntEnum):
    amount_quarter_perp_3 = auto()
    amount_quarter_spot_3 = auto()
    amount_quarter_spot_4 = auto()
    no_group = auto()


class Category(IntEnum):
    liquidity_1d = auto()
    momentum_1d = auto()
    pv_1d = auto()
    volatility_1d = auto()
    imbalance_1d = auto()
    technical_1d = auto()

    imbalance_hf_1d = auto()
    liquidity_hf_1d = auto()
    momentum_hf_1d = auto()
    pv_hf_1d = auto()
    volatility_hf_1d = auto()

    liquidity_1h = auto()
    momentum_1h = auto()
    pv_1h = auto()
    volatility_1h = auto()
    game_1h = auto()
    imbalance_1h = auto()

    imbalance_hf_1h = auto()
    liquidity_hf_1h = auto()
    momentum_hf_1h = auto()
    pv_hf_1h = auto()
    volatility_hf_1h = auto()

    development_1d = auto()
    development_1h = auto()

    disagree_contract = auto()
    game_contract = auto()
    liquidity_contract = auto()
    momentum_contract = auto()
    turnover_contract = auto()
    volatility_contract = auto()


@dataclass
class DataSources:
    kline: "DataSource"
    group: "DataSource"


@dataclass
class DataSource:
    exchange: "Exchange"
    universe: "Universe"
    freq: Union[Literal["1h", "1d"], None] = None


class Exchange(IntEnum):
    Okx = auto()
    Binance = auto()
    Crossover = auto()


class Universe(IntEnum):
    spot = auto()
    perp = auto()


class Group(IntEnum):
    BIG = 0
    MIDDLE = 1
    SMALL = 2

    ALL = 3


@dataclass
class Alpha:
    category: "Category"
    alpha: str
    aggregations: dict = field(default_factory=dict)
    exchange: Exchange = Exchange.Binance
    universe: Universe = Universe.spot
    freq: Optional[Literal["1h", "1d"]] = "1d"

    @property
    def path(self) -> Path:
        return Path(f"Alphas/{self.freq}/{self.category.name}/{self.alpha}.csv")

    @property
    def name(self) -> str:
        if not self.aggregations:
            return self.alpha
        else:
            key, value = next(iter(self.aggregations.items()))
            return f"{self.alpha}_{key}{value}"


@dataclass
class TrainTestSpans:
    train_start: datetime.datetime
    train_end: datetime.datetime

    valid_start: Optional[datetime.datetime] = None
    valid_end: Optional[datetime.datetime] = None
    test_start: Optional[datetime.datetime] = None
    test_end: Optional[datetime.datetime] = None

    allow_overlap: bool = True

    def __post_init__(self):
        """Validate date ranges and overlaps."""
        self._validate_date_order()
        if not self.allow_overlap:
            self._validate_no_overlap()

    def _validate_date_order(self) -> None:
        """Ensure dates are in chronological order within each period."""
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")

        if self.valid_start and self.valid_end and self.valid_start >= self.valid_end:
            raise ValueError("valid_start must be before valid_end")

        if self.test_start and self.test_end and self.test_start >= self.test_end:
            raise ValueError("test_start must be before test_end")

    def _validate_no_overlap(self) -> None:
        """Ensure no overlap between periods when allow_overlap=False."""
        if not all([self.valid_start, self.valid_end, self.test_start, self.test_end]):
            raise ValueError(
                "All date ranges must be specified when allow_overlap=False"
            )

        if self.train_end > self.valid_start:
            raise ValueError("Training period overlaps with validation period")

        if self.valid_end > self.test_start:
            raise ValueError("Validation period overlaps with test period")

    @property
    def has_validation(self) -> bool:
        """Check if validation period is defined."""
        return self.valid_start is not None and self.valid_end is not None

    @property
    def has_test(self) -> bool:
        """Check if test period is defined."""
        return self.test_start is not None and self.test_end is not None
