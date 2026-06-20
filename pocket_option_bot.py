# ============================================================
# POCKET OPTION TRADING BOT — Google Colab Backend  v2.0
# Run each cell block in order inside Google Colab
# ============================================================

# ─────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────
"""
!pip install websocket-client flask flask-cors pyngrok pandas numpy ta requests
"""

# ─────────────────────────────────────────────────────────────
# CELL 2 — Imports & Config
# ─────────────────────────────────────────────────────────────
import websocket
import json
import threading
import time
import queue
import datetime
import logging
from collections import deque

import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD as MACDIndicator
from ta.volatility import BollingerBands, AverageTrueRange

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pyngrok import ngrok, conf

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s — %(message)s")
log = logging.getLogger("PocketBot")

# ─────────────────────────────────────────────────────────────
# CELL 3 — Signal Engine (Technical Analysis)
# ─────────────────────────────────────────────────────────────

# ── v2.0 Changes ──────────────────────────────────────────────
# • Confidence threshold raised 50% → 70%
# • Added ATR (Average True Range) indicator + vote
# • Added Volume analysis indicator + vote
# • Scoring is now weighted (heavier indicators = more weight)
# ──────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 50   # ← lowered to 50 for more signals with pattern analysis

# ──────────────────────────────────────────────────────────────
# Candlestick Pattern Recognition
# ──────────────────────────────────────────────────────────────

def detect_bullish_patterns(candles: list) -> tuple:
    """Detect bullish candlestick patterns. Returns (pattern_name, strength 0-2)."""
    if len(candles) < 3:
        return None, 0
    
    # Last 3 candles
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    o1, h1, l1, close1 = c1['open'], c1['high'], c1['low'], c1['close']
    o2, h2, l2, close2 = c2['open'], c2['high'], c2['low'], c2['close']
    o3, h3, l3, close3 = c3['open'], c3['high'], c3['low'], c3['close']
    
    patterns = []
    
    # Hammer: Long lower wick, small body at top
    if close3 > o3:  # Green candle
        wick_lower = min(o3, close3) - l3
        body = abs(close3 - o3)
        wick_upper = h3 - max(o3, close3)
        if wick_lower > body * 2 and wick_upper < body * 0.5:
            patterns.append(("Hammer", 2))
    
    # Bullish Engulfing: Previous red, current green fully encompasses
    if close2 < o2 and close3 > o3:  # Previous red, current green
        if close3 > o2 and o3 < close2:
            patterns.append(("Bullish Engulfing", 2))
    
    # Three White Soldiers: 3 consecutive green candles, each closing higher
    if close1 < o1 or close2 < o2 or close3 < o3:
        pass  # Not all green
    else:
        if close1 < close2 < close3:
            patterns.append(("Three White Soldiers", 2))
    
    # Piercing Line: Red candle, green candle closes above midpoint of red
    if close2 < o2 and close3 > o3:  # Prev red, curr green
        midpoint = (o2 + close2) / 2
        if close3 > midpoint and o3 < close2:
            patterns.append(("Piercing Line", 1))
    
    # Morning Star: Red, small body, green (trend reversal)
    if len(candles) >= 3:
        if close1 < o1 and close3 > o3:  # Red then green
            body2 = abs(close2 - o2)
            body1 = abs(close1 - o1)
            if body2 < body1 * 0.5:  # Small middle body
                patterns.append(("Morning Star", 2))
    
    return patterns[0] if patterns else (None, 0)


def detect_bearish_patterns(candles: list) -> tuple:
    """Detect bearish candlestick patterns. Returns (pattern_name, strength 0-2)."""
    if len(candles) < 3:
        return None, 0
    
    # Last 3 candles
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    o1, h1, l1, close1 = c1['open'], c1['high'], c1['low'], c1['close']
    o2, h2, l2, close2 = c2['open'], c2['high'], c2['low'], c2['close']
    o3, h3, l3, close3 = c3['open'], c3['high'], c3['low'], c3['close']
    
    patterns = []
    
    # Inverted Hammer/Shooting Star: Long upper wick, small body at bottom
    if close3 < o3:  # Red candle
        wick_upper = h3 - max(o3, close3)
        body = abs(close3 - o3)
        wick_lower = min(o3, close3) - l3
        if wick_upper > body * 2 and wick_lower < body * 0.5:
            patterns.append(("Shooting Star", 2))
    
    # Bearish Engulfing: Previous green, current red fully encompasses
    if close2 > o2 and close3 < o3:  # Previous green, current red
        if close3 < o2 and o3 > close2:
            patterns.append(("Bearish Engulfing", 2))
    
    # Three Black Crows: 3 consecutive red candles, each closing lower
    if close1 > o1 or close2 > o2 or close3 > o3:
        pass  # Not all red
    else:
        if close1 > close2 > close3:
            patterns.append(("Three Black Crows", 2))
    
    # Dark Cloud Cover: Green candle, red candle closes below midpoint
    if close2 > o2 and close3 < o3:  # Prev green, curr red
        midpoint = (o2 + close2) / 2
        if close3 < midpoint and o3 > close2:
            patterns.append(("Dark Cloud Cover", 1))
    
    # Evening Star: Green, small body, red (trend reversal)
    if len(candles) >= 3:
        if close1 > o1 and close3 < o3:  # Green then red
            body2 = abs(close2 - o2)
            body1 = abs(close1 - o1)
            if body2 < body1 * 0.5:  # Small middle body
                patterns.append(("Evening Star", 2))
    
    return patterns[0] if patterns else (None, 0)


def generate_signal(candles: list) -> dict:
    """
    Multi-indicator signal generator (v2.0).
    Indicators: RSI, EMA crossover, MACD, Bollinger Bands,
                Stochastic, ATR (volatility filter), Volume (momentum).
    Returns direction (UP / DOWN / WAIT) + confidence score (0–100).
    """
    if len(candles) < 30:
        return {
            "direction": "WAIT",
            "confidence": 0,
            "reason": "Collecting data...",
            "indicators": {}
        }

    df = pd.DataFrame(candles).tail(100)
    closes  = df["close"].astype(float)
    highs   = df["high"].astype(float)
    lows    = df["low"].astype(float)

    # Volume column — use zeros if not available (OTC feeds often omit it)
    if "volume" in df.columns:
        volumes = df["volume"].astype(float)
        has_volume = volumes.sum() > 0
    else:
        volumes   = pd.Series(np.zeros(len(df)), index=df.index)
        has_volume = False

    votes   = []   # (weight, direction_int) — positive = bullish
    reasons = []

    # ── CANDLESTICK PATTERNS (NEW - Higher priority) ──────────
    bullish_pattern, bull_strength = detect_bullish_patterns(list(candles))
    bearish_pattern, bear_strength = detect_bearish_patterns(list(candles))
    
    if bullish_pattern:
        votes.append(2 if bull_strength == 2 else 1)
        reasons.append(f"Candlestick: {bullish_pattern} ↑")
    elif bearish_pattern:
        votes.append(-2 if bear_strength == 2 else -1)
        reasons.append(f"Candlestick: {bearish_pattern} ↓")
    
    # ── RSI ───────────────────────────────────────────────────
    rsi_val = RSIIndicator(closes, window=14).rsi().iloc[-1]
    if rsi_val < 30:
        votes.append(2);  reasons.append(f"RSI oversold ({rsi_val:.1f}) ↑")
    elif rsi_val > 70:
        votes.append(-2); reasons.append(f"RSI overbought ({rsi_val:.1f}) ↓")
    elif rsi_val < 45:
        votes.append(1);  reasons.append(f"RSI bearish zone ({rsi_val:.1f})")
    elif rsi_val > 55:
        votes.append(-1); reasons.append(f"RSI bullish zone ({rsi_val:.1f})")
    else:
        votes.append(0)

    # ── EMA Crossover (5 / 13 / 50) ──────────────────────────
    ema5  = EMAIndicator(closes, window=5).ema_indicator()
    ema13 = EMAIndicator(closes, window=13).ema_indicator()
    ema50 = EMAIndicator(closes, window=50).ema_indicator()

    if ema5.iloc[-1] > ema13.iloc[-1] and ema5.iloc[-2] <= ema13.iloc[-2]:
        votes.append(2);  reasons.append("EMA5 crossed ↑ EMA13 (bullish)")
    elif ema5.iloc[-1] < ema13.iloc[-1] and ema5.iloc[-2] >= ema13.iloc[-2]:
        votes.append(-2); reasons.append("EMA5 crossed ↓ EMA13 (bearish)")
    elif ema5.iloc[-1] > ema13.iloc[-1]:
        votes.append(1);  reasons.append("EMA5 above EMA13 (bullish)")
    else:
        votes.append(-1); reasons.append("EMA5 below EMA13 (bearish)")

    last_close = closes.iloc[-1]
    if last_close > ema50.iloc[-1]:
        votes.append(1);  reasons.append("Price above EMA50 (uptrend)")
    else:
        votes.append(-1); reasons.append("Price below EMA50 (downtrend)")

    # ── MACD ──────────────────────────────────────────────────
    macd_obj  = MACDIndicator(closes, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_obj.macd().iloc[-1]
    macd_sig  = macd_obj.macd_signal().iloc[-1]
    macd_hist = macd_obj.macd_diff().iloc[-1]
    macd_prev = macd_obj.macd_diff().iloc[-2]

    if macd_hist > 0 and macd_prev <= 0:
        votes.append(2);  reasons.append("MACD histogram turned bullish ↑")
    elif macd_hist < 0 and macd_prev >= 0:
        votes.append(-2); reasons.append("MACD histogram turned bearish ↓")
    elif macd_line > macd_sig:
        votes.append(1);  reasons.append("MACD above signal (bullish)")
    else:
        votes.append(-1); reasons.append("MACD below signal (bearish)")

    # ── Bollinger Bands ───────────────────────────────────────
    bb       = BollingerBands(closes, window=20, window_dev=2)
    bb_lower = bb.bollinger_lband().iloc[-1]
    bb_upper = bb.bollinger_hband().iloc[-1]
    bb_mid   = bb.bollinger_mavg().iloc[-1]
    bb_pct   = bb.bollinger_pband().iloc[-1]

    if bb_pct <= 0.05:
        votes.append(2);  reasons.append("Price at lower Bollinger Band ↑")
    elif bb_pct >= 0.95:
        votes.append(-2); reasons.append("Price at upper Bollinger Band ↓")
    elif bb_pct < 0.4:
        votes.append(1)
    elif bb_pct > 0.6:
        votes.append(-1)
    else:
        votes.append(0)

    # ── Stochastic ────────────────────────────────────────────
    stoch   = StochasticOscillator(highs, lows, closes, window=14, smooth_window=3)
    stoch_k = stoch.stoch().iloc[-1]
    stoch_d = stoch.stoch_signal().iloc[-1]

    if stoch_k < 20 and stoch_k > stoch_d:
        votes.append(2);  reasons.append(f"Stochastic oversold + K>D ({stoch_k:.1f}) ↑")
    elif stoch_k > 80 and stoch_k < stoch_d:
        votes.append(-2); reasons.append(f"Stochastic overbought + K<D ({stoch_k:.1f}) ↓")
    elif stoch_k < 20:
        votes.append(1);  reasons.append(f"Stochastic oversold ({stoch_k:.1f})")
    elif stoch_k > 80:
        votes.append(-1); reasons.append(f"Stochastic overbought ({stoch_k:.1f})")
    else:
        votes.append(0)

    # ── ATR — Volatility Filter (NEW in v2.0) ─────────────────
    # ATR tells us if market is "tradeable" (enough movement).
    # We use ATR relative to price (normalised ATR %).
    atr_indicator = AverageTrueRange(highs, lows, closes, window=14)
    atr_val       = atr_indicator.average_true_range().iloc[-1]
    atr_pct       = (atr_val / last_close) * 100 if last_close > 0 else 0

    # ATR doesn't vote for direction but boosts confidence when market is active.
    # Weak ATR (<0.02%) → reduce confidence later; strong ATR → normal.
    atr_active = atr_pct >= 0.02   # threshold: at least 0.02% range per candle

    if atr_pct >= 0.05:
        reasons.append(f"ATR high volatility ({atr_pct:.3f}%) — strong trend")
    elif atr_pct >= 0.02:
        reasons.append(f"ATR normal volatility ({atr_pct:.3f}%)")
    else:
        reasons.append(f"ATR low volatility ({atr_pct:.4f}%) — caution")

    # ── Volume Analysis (NEW in v2.0) ─────────────────────────
    vol_ratio = 1.0    # neutral default when no volume data
    vol_surge = False

    if has_volume and len(volumes) >= 20:
        avg_vol   = volumes.iloc[-20:-1].mean()
        last_vol  = volumes.iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        if vol_ratio >= 1.5:
            vol_surge = True
            # Volume surge confirms the direction of price movement
            price_change = closes.iloc[-1] - closes.iloc[-2]
            if price_change > 0:
                votes.append(2);  reasons.append(f"Volume surge {vol_ratio:.1f}x avg (bullish confirm) ↑")
            else:
                votes.append(-2); reasons.append(f"Volume surge {vol_ratio:.1f}x avg (bearish confirm) ↓")
        elif vol_ratio >= 1.2:
            votes.append(1 if closes.iloc[-1] > closes.iloc[-2] else -1)
            reasons.append(f"Volume above avg ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.7:
            votes.append(0)
            reasons.append(f"Volume below avg ({vol_ratio:.1f}x) — weak signal")
        else:
            votes.append(0)

    # ── Score ──────────────────────────────────────────────────
    total     = sum(votes)
    max_score = sum(abs(v) for v in votes) or 1
    raw_conf  = abs(total) / max_score * 100

    # Pattern boost: if a strong pattern was detected, boost confidence
    if bullish_pattern and bull_strength == 2:
        raw_conf = min(100, raw_conf + 20)
    elif bearish_pattern and bear_strength == 2:
        raw_conf = min(100, raw_conf + 20)

    # ATR penalty: if volatility too low, reduce confidence by up to 10 points
    if not atr_active:
        raw_conf = max(0, raw_conf - 10)

    confidence = round(raw_conf, 1)

    if total > 0:
        direction = "UP"
    elif total < 0:
        direction = "DOWN"
    else:
        direction = "WAIT"

    # ── Apply threshold (PATTERN signals override low thresholds) ────────────
    # If we have a strong candlestick pattern, lower the threshold
    if bullish_pattern or bearish_pattern:
        pattern_threshold = 35  # Patterns only need 35% confidence
    else:
        pattern_threshold = CONFIDENCE_THRESHOLD  # Default 50%
    
    if confidence < pattern_threshold and direction != "WAIT":
        direction = "WAIT"

    return {
        "direction":    direction,
        "confidence":   confidence,
        "reasons":      reasons[:5],       # top 5 reasons
        "indicators": {
            "rsi":      round(float(rsi_val), 2),
            "ema5":     round(float(ema5.iloc[-1]), 6),
            "ema13":    round(float(ema13.iloc[-1]), 6),
            "macd":     round(float(macd_line), 6),
            "macd_sig": round(float(macd_sig), 6),
            "stoch_k":  round(float(stoch_k), 2),
            "bb_pct":   round(float(bb_pct), 4),
            "atr":      round(float(atr_val), 6),
            "atr_pct":  round(float(atr_pct), 4),
            "vol_ratio": round(float(vol_ratio), 2),
            "vol_surge": vol_surge,
        },
        "last_close":   round(float(last_close), 6),
        "candle_count": len(candles),
    }


# ─────────────────────────────────────────────────────────────
# CELL 4 — Pocket Option WebSocket Client
# ─────────────────────────────────────────────────────────────

# ── v2.0 Changes ──────────────────────────────────────────────
# • Exponential back-off reconnection (1s → 2s → 4s … max 60s)
# • Max reconnect attempts cap (configurable, default 20)
# • Signal cooldown: won't re-emit the same direction
#   within SIGNAL_COOLDOWN_SECONDS (default 30s)
# • Thread-safe reconnect lock (prevents double reconnects)
# ──────────────────────────────────────────────────────────────

SIGNAL_COOLDOWN_SECONDS = 15   # seconds between same-direction signals (reduced for more frequent signals)
MAX_RECONNECT_ATTEMPTS  = 20   # give up after this many consecutive failures


class PocketOptionBot:
    WS_URL = "wss://api.po.market/socket.io/?EIO=4&transport=websocket"

    def __init__(self, ssid: str, asset: str = "EURUSD_otc",
                 timeframe: int = 60, is_demo: bool = True):
        self.ssid      = ssid
        self.asset     = asset
        self.timeframe = timeframe
        self.is_demo   = is_demo

        self.candles: deque  = deque(maxlen=200)
        self.signal: dict    = {"direction": "WAIT", "confidence": 0,
                                 "reasons": [], "indicators": {}}
        self.last_price: float  = 0.0
        self.status: str        = "Disconnected"
        self.ws                 = None
        self._thread            = None
        self._ping_thread       = None
        self._sid               = None
        self._authed            = False
        self._running           = False

        # ── Reconnect state ─────────────────────────────────────
        self._reconnect_attempts  = 0
        self._reconnect_lock      = threading.Lock()
        self._backoff_base        = 1    # seconds
        self._backoff_max         = 60   # cap

        # ── Signal cooldown state ────────────────────────────────
        self._last_signal_time: dict = {}   # {direction: datetime}

    # ── Back-off helper ────────────────────────────────────────

    def _backoff_delay(self) -> float:
        """Exponential back-off: 1, 2, 4, 8 … up to _backoff_max seconds."""
        delay = min(self._backoff_base * (2 ** self._reconnect_attempts),
                    self._backoff_max)
        return delay

    # ── Signal cooldown check ──────────────────────────────────

    def _is_cooled_down(self, direction: str) -> bool:
        """Return True if enough time has passed since the last signal of this direction."""
        last = self._last_signal_time.get(direction)
        if last is None:
            return True
        elapsed = (datetime.datetime.utcnow() - last).total_seconds()
        return elapsed >= SIGNAL_COOLDOWN_SECONDS

    def _record_signal(self, direction: str):
        self._last_signal_time[direction] = datetime.datetime.utcnow()

    # ── Connection helpers ─────────────────────────────────────

    def _send(self, msg: str):
        if self.ws:
            try:
                self.ws.send(msg)
                log.debug(f"→ {msg[:120]}")
            except Exception as e:
                log.warning(f"Send error: {e}")

    def _ping_loop(self):
        """Keep socket alive with Socket.IO pings."""
        while self._running:
            time.sleep(20)
            if self.ws and self._authed:
                self._send("2")

    # ── WebSocket callbacks ────────────────────────────────────

    def on_open(self, ws):
        log.info("WebSocket opened")
        self.status = "Connecting..."
        # Reset reconnect counter on successful open
        self._reconnect_attempts = 0

    def on_message(self, ws, raw: str):
        log.debug(f"← {raw[:200]}")

        if raw.startswith("0"):
            try:
                data = json.loads(raw[1:])
                self._sid = data.get("sid")
            except Exception:
                pass
            self._send("40")
            return

        if raw == "40" or raw.startswith("40{"):
            log.info("Namespace connected — authenticating …")
            demo_flag = 1 if self.is_demo else 0
            self._send(f'42["auth",{{"session":"{self.ssid}","isDemo":{demo_flag}}}]')
            return

        if raw == "3":
            return

        if not raw.startswith("42"):
            return

        try:
            payload = json.loads(raw[2:])
        except Exception:
            return

        event = payload[0] if isinstance(payload, list) else None
        data  = payload[1] if len(payload) > 1 else {}

        if event in ("successauth", "successAuth"):
            self._authed = True
            self.status  = "Authenticated"
            log.info("✓ Authenticated — subscribing to asset …")
            self._subscribe()
            return

        if event in ("failureauth", "failureAuth"):
            self.status = "Auth Failed — check SSID"
            log.error("✗ Authentication failed")
            return

        if event in ("candles", "history", "candle", "updateStream"):
            self._handle_candle_data(data)

        if event in ("tick", "price"):
            price = data.get("price") or data.get("value")
            if price:
                self.last_price = float(price)

    def _subscribe(self):
        """Request candle history + live stream."""
        self._send('42' + json.dumps([
            "sendMessage",
            {
                "name": "get_candles",
                "data": {
                    "asset":  self.asset,
                    "period": self.timeframe,
                    "offset": 0,
                    "count":  100
                }
            }
        ]))
        self._send('42' + json.dumps([
            "subscribeSymbol",
            {"asset": self.asset, "period": self.timeframe}
        ]))
        self.status = f"Live — {self.asset} / {self.timeframe}s"
        log.info(f"Subscribed: {self.asset} @ {self.timeframe}s")

    def _handle_candle_data(self, data):
        """Parse incoming candle data and update signal."""
        raw_candles = None

        if isinstance(data, list):
            raw_candles = data
        elif isinstance(data, dict):
            raw_candles = (data.get("candles") or
                           data.get("data")    or
                           data.get("history"))
            if not raw_candles and "close" in data:
                raw_candles = [data]

        if not raw_candles:
            return

        for c in raw_candles:
            try:
                candle = {
                    "time":   c.get("time")   or c.get("t") or c.get("timestamp"),
                    "open":   float(c.get("open")   or c.get("o") or 0),
                    "high":   float(c.get("high")   or c.get("h") or 0),
                    "low":    float(c.get("low")    or c.get("l") or 0),
                    "close":  float(c.get("close")  or c.get("c") or 0),
                    "volume": float(c.get("volume") or c.get("v") or 0),
                }
                if candle["close"] > 0:
                    self.candles.append(candle)
                    self.last_price = candle["close"]
            except Exception:
                pass

        if len(self.candles) >= 5:
            new_signal = generate_signal(list(self.candles))

            # ── Cooldown gate ──────────────────────────────────
            if new_signal["direction"] != "WAIT":
                if self._is_cooled_down(new_signal["direction"]):
                    self.signal = new_signal
                    self._record_signal(new_signal["direction"])
                    log.info(f"✅ New signal: {new_signal['direction']} "
                             f"({new_signal['confidence']}%) — cooldown reset")
                else:
                    # Update indicators silently but don't re-emit signal
                    self.signal["indicators"]   = new_signal["indicators"]
                    self.signal["last_close"]   = new_signal["last_close"]
                    self.signal["candle_count"] = new_signal["candle_count"]
                    remaining = SIGNAL_COOLDOWN_SECONDS - (
                        datetime.datetime.utcnow() -
                        self._last_signal_time[new_signal["direction"]]
                    ).total_seconds()
                    log.debug(f"⏳ Cooldown active for {new_signal['direction']} "
                              f"— {remaining:.0f}s remaining")
            else:
                self.signal = new_signal

        log.info(f"Candles: {len(self.candles)} | Signal: "
                 f"{self.signal['direction']} ({self.signal['confidence']}%)")

    def on_error(self, ws, error):
        log.error(f"WebSocket error: {error}")
        self.status = f"Error: {str(error)[:60]}"

    def on_close(self, ws, code, msg):
        log.warning(f"WebSocket closed [{code}]: {msg}")
        self.status  = "Disconnected"
        self._authed = False

        if not self._running:
            return

        # ── Exponential back-off reconnect ──────────────────────
        with self._reconnect_lock:
            if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                log.error(f"Max reconnect attempts ({MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
                self.status  = "Reconnect failed — restart manually"
                self._running = False
                return

            delay = self._backoff_delay()
            self._reconnect_attempts += 1
            log.info(f"Reconnecting in {delay:.0f}s "
                     f"(attempt {self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}) …")
            self.status = f"Reconnecting… ({self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS})"

        time.sleep(delay)
        if self._running:
            self.start(reset_counter=False)   # preserve attempt counter

    # ── Public API ─────────────────────────────────────────────

    def start(self, reset_counter: bool = True):
        self._running = True
        if reset_counter:
            self._reconnect_attempts = 0

        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open    = self.on_open,
            on_message = self.on_message,
            on_error   = self.on_error,
            on_close   = self.on_close,
            header     = {
                "User-Agent": "Mozilla/5.0",
                "Origin":     "https://po.market",
            }
        )
        self._thread = threading.Thread(
            target=self.ws.run_forever, kwargs={"ping_interval": 25}, daemon=True
        )
        self._thread.start()

        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()
        log.info("Bot started")

    def stop(self):
        self._running = False
        if self.ws:
            self.ws.close()
        log.info("Bot stopped")

    def change_asset(self, asset: str, timeframe: int = None):
        if timeframe:
            self.timeframe = timeframe
        self.asset = asset
        self.candles.clear()
        self.signal = {"direction": "WAIT", "confidence": 0,
                       "reasons": [], "indicators": {}}
        self._last_signal_time = {}   # reset cooldowns on asset change
        if self._authed:
            self._subscribe()


# ─────────────────────────────────────────────────────────────
# CELL 5 — Flask REST API
# ─────────────────────────────────────────────────────────────
import os

app  = Flask(__name__)
CORS(app)
bot: PocketOptionBot = None

# ── HTML Dashboard content (embedded) ──────────────────────────
DASHBOARD_HTML = None

def load_dashboard():
    """Load the dashboard HTML from file if available, fallback to minimal dashboard."""
    global DASHBOARD_HTML
    dashboard_path = os.path.join(os.path.dirname(__file__), 'trading_dashboard.html')
    if os.path.exists(dashboard_path):
        try:\


            
            with open(dashboard_path, 'r', encoding='utf-8') as f:
                DASHBOARD_HTML = f.read()
                print(f"✅ Dashboard HTML loaded: {len(DASHBOARD_HTML)} bytes")
        except Exception as e:
            print(f"❌ Could not load dashboard HTML: {e}")
            log.warning(f"Could not load dashboard HTML: {e}")
    else:
        print(f"❌ Dashboard file not found: {dashboard_path}")
    return DASHBOARD_HTML

# Load dashboard immediately on startup
load_dashboard()

ASSETS = [
    # Forex OTC
    "EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDUSD_otc",
    "USDCAD_otc", "EURGBP_otc", "EURJPY_otc", "GBPJPY_otc",
    "USDCHF_otc", "NZDUSD_otc",
    # Stocks OTC
    "#AAPL_otc",  "#TSLA_otc",  "#AMZN_otc",  "#GOOGL_otc",
    "#MSFT_otc",  "#META_otc",  "#NFLX_otc",  "#NVDA_otc",
    # Commodities OTC
    "XAUUSD_otc", "XAGUSD_otc",
    # Crypto OTC
    "BTCUSD_otc", "ETHUSD_otc", "LTCUSD_otc", "XRPUSD_otc",
]

TIMEFRAMES = {
    "5s":  5,   "10s": 10,  "15s": 15,  "30s": 30,
    "1m":  60,  "2m":  120, "5m":  300, "15m": 900,
}

signal_history  = deque(maxlen=100)
last_signal_dir = "WAIT"

# ── Trade outcome tracking (win rate) ─────────────────────────
# Outcomes are recorded via POST /outcome endpoint
# { "signal_id": N, "result": "WIN" | "LOSS" }
outcome_store = {}      # signal_id → result


@app.route("/")
def dashboard():
    """Serve the trading dashboard."""
    try:
        if DASHBOARD_HTML:
            return Response(DASHBOARD_HTML, mimetype="text/html")
        else:
            return Response("<h1>PocketBot Dashboard</h1><p>Dashboard HTML not loaded. DASHBOARD_HTML is None. Check trading_dashboard.html exists in bot directory.</p>", mimetype="text/html")
    except Exception as e:
        return Response(f"<h1>Error</h1><p>{str(e)}</p>", mimetype="text/html")


@app.route("/status")
def status():
    if not bot:
        return jsonify({"status": "Not started", "connected": False})
    return jsonify({
        "status":              bot.status,
        "connected":           bot._authed,
        "asset":               bot.asset,
        "timeframe":           bot.timeframe,
        "candles":             len(bot.candles),
        "price":               bot.last_price,
        "reconnect_attempts":  bot._reconnect_attempts,
        "cooldown_remaining":  _cooldown_remaining(),
    })


def _cooldown_remaining() -> float:
    """Seconds until the active direction cooldown expires (0 if none)."""
    if not bot:
        return 0
    sig = bot.signal.get("direction", "WAIT")
    if sig == "WAIT":
        return 0
    last = bot._last_signal_time.get(sig)
    if not last:
        return 0
    elapsed = (datetime.datetime.utcnow() - last).total_seconds()
    return max(0, round(SIGNAL_COOLDOWN_SECONDS - elapsed, 1))


@app.route("/signal")
def get_signal():
    global last_signal_dir

    if not bot:
        return jsonify({"direction": "WAIT", "confidence": 0,
                        "message": "Bot not started"})

    sig = dict(bot.signal)
    sig["asset"]              = bot.asset
    sig["timeframe"]          = bot.timeframe
    sig["price"]              = bot.last_price
    sig["timestamp"]          = datetime.datetime.utcnow().isoformat() + "Z"
    sig["status"]             = bot.status
    sig["cooldown_remaining"] = _cooldown_remaining()

    if sig["direction"] != "WAIT" and sig["direction"] != last_signal_dir:
        entry = {
            "id":         len(signal_history) + 1,
            "direction":  sig["direction"],
            "confidence": sig["confidence"],
            "asset":      sig["asset"],
            "price":      sig["price"],
            "time":       sig["timestamp"],
            "result":     None,
        }
        signal_history.appendleft(entry)
        last_signal_dir = sig["direction"]

    return jsonify(sig)


@app.route("/history")
def get_history():
    hist = list(signal_history)
    # Attach outcomes
    for entry in hist:
        entry["result"] = outcome_store.get(entry.get("id"))
    # Win rate stats
    decided = [e for e in hist if e["result"] in ("WIN", "LOSS")]
    wins    = sum(1 for e in decided if e["result"] == "WIN")
    stats   = {
        "total":    len(hist),
        "decided":  len(decided),
        "wins":     wins,
        "losses":   len(decided) - wins,
        "win_rate": round(wins / len(decided) * 100, 1) if decided else None,
    }
    return jsonify({"signals": hist, "stats": stats})


@app.route("/outcome", methods=["POST"])
def record_outcome():
    """Record the result of a signal trade. Body: {signal_id, result}"""
    body      = request.get_json(force=True) or {}
    signal_id = body.get("signal_id")
    result    = body.get("result", "").upper()
    if result not in ("WIN", "LOSS"):
        return jsonify({"error": "result must be WIN or LOSS"}), 400
    outcome_store[signal_id] = result
    return jsonify({"message": f"Signal {signal_id} recorded as {result}"})


@app.route("/start", methods=["POST"])
def start_bot():
    global bot
    body      = request.get_json(force=True) or {}
    ssid      = body.get("ssid", "").strip()
    asset     = body.get("asset", "EURUSD_otc")
    tf_label  = body.get("timeframe", "1m")
    is_demo   = body.get("demo", True)
    timeframe = TIMEFRAMES.get(tf_label, 60)

    if not ssid:
        return jsonify({"error": "SSID is required"}), 400

    if bot:
        bot.stop()
        time.sleep(1)

    bot = PocketOptionBot(ssid=ssid, asset=asset,
                          timeframe=timeframe, is_demo=is_demo)
    bot.start()
    return jsonify({"message": "Bot started", "asset": asset,
                    "timeframe": tf_label, "demo": is_demo})


@app.route("/stop", methods=["POST"])
def stop_bot():
    global bot
    if bot:
        bot.stop()
        bot = None
    return jsonify({"message": "Bot stopped"})


@app.route("/change", methods=["POST"])
def change_asset():
    global bot
    if not bot:
        return jsonify({"error": "Bot not running"}), 400

    body      = request.get_json(force=True) or {}
    asset     = body.get("asset", bot.asset)
    tf_label  = body.get("timeframe", "1m")
    timeframe = TIMEFRAMES.get(tf_label, 60)
    bot.change_asset(asset, timeframe)
    return jsonify({"message": f"Changed to {asset} @ {tf_label}"})


@app.route("/assets")
def get_assets():
    return jsonify(ASSETS)


@app.route("/timeframes")
def get_timeframes():
    return jsonify(list(TIMEFRAMES.keys()))


@app.route("/config")
def get_config():
    """Return current bot configuration constants."""
    return jsonify({
        "confidence_threshold":    CONFIDENCE_THRESHOLD,
        "signal_cooldown_seconds": SIGNAL_COOLDOWN_SECONDS,
        "max_reconnect_attempts":  MAX_RECONNECT_ATTEMPTS,
    })


# ─────────────────────────────────────────────────────────────
# CELL 6 — Launch (run this last in Colab)
# ─────────────────────────────────────────────────────────────

def launch(ngrok_token: str = "", port: int = 5000, use_ngrok: bool = True):
    """
    Start Flask server with optional ngrok tunnel.
    If ngrok_token is provided, creates a public URL via ngrok.
    """
    load_dashboard()
    
    print("\n" + "═" * 70)
    if use_ngrok and ngrok_token:
        conf.get_default().auth_token = ngrok_token
        try:
            public_url = ngrok.connect(port, "http")
            print(f"  🚀  NGROK PUBLIC URL  →  {public_url}")
            print(f"  📡  LOCAL URL         →  http://localhost:{port}")
            print("═" * 70)
            print("\n  ✅ Use the PUBLIC URL to access your bot from anywhere!")
        except Exception as e:
            print(f"  ⚠️  Ngrok connection failed: {e}")
            print(f"  📡  LOCAL URL         →  http://localhost:{port}")
            print("═" * 70)
    else:
        print(f"  📡  LOCAL URL         →  http://localhost:{port}")
        print("═" * 70)
    
    print("\n  API endpoints:")
    print(f"    GET  /              — trading dashboard (web UI)")
    print(f"    POST /start         — start bot  (body: ssid, asset, timeframe, demo)")
    print(f"    GET  /signal        — live signal")
    print(f"    GET  /status        — connection status")
    print(f"    GET  /history       — signal history + win rate")
    print(f"    POST /outcome       — record trade result")
    print(f"    POST /change        — switch asset/timeframe")
    print(f"    POST /stop          — stop bot\n")
    print("  ✨ v2.1 FEATURES (Enhanced):")
    print(f"    • Candlestick Pattern Recognition (8+ patterns detected)")
    print(f"    • Confidence threshold: 50% (patterns: 35%)")
    print(f"    • Signal cooldown: {SIGNAL_COOLDOWN_SECONDS}s between signals")
    print(f"    • Multi-indicator: RSI, EMA, MACD, Stochastic, BB, ATR, Volume")
    print(f"    • Auto-reconnect with exponential backoff\n")

    app.run(host="0.0.0.0", port=port, debug=False)


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    # Use ngrok with your authtoken
    NGROK_TOKEN = "3EcXK5KqKQtZrF8B5GJmr9cnT2t_6eX2vuYbq71xpancEHkcL"
    launch(ngrok_token=NGROK_TOKEN, port=5000, use_ngrok=True)
