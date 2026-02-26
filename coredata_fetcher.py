"""
Market data fetcher with multi-exchange support and robust error handling.
Uses CCXT for exchange connectivity with connection pooling and rate limiting.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"  # Note: FTX may not be operational, included for demonstration
    BYBIT = "bybit"


@dataclass
class MarketData:
    """Structured market data container."""
    symbol: str
    exchange: ExchangeType
    timestamp