from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class PortfolioSignal:
    symbol: str
    signal_type: str
    strength: Optional[float] = None
    timestamp: Optional[str] = None
    date: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortfolioSignal":
        return PortfolioSignal(
            symbol=str(d.get("symbol") or ""),
            signal_type=str(d.get("signalType") or d.get("signal_type") or d.get("type") or "").upper(),
            strength=(None if d.get("strength") is None else float(d.get("strength"))),
            timestamp=(d.get("signalTimestamp") or d.get("signal_timestamp")),
            date=(d.get("signalDate") or d.get("signal_date")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        base = f"{self.symbol} {self.signal_type}"
        if self.strength is not None:
            base += f" {self.strength}"
        return base


@dataclass
class PortfolioWeight:
    symbol: str
    final_weight: Optional[float] = None
    raw_weight: Optional[float] = None
    timestamp: Optional[str] = None
    date: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortfolioWeight":
        return PortfolioWeight(
            symbol=str(d.get("symbol") or ""),
            final_weight=(None if d.get("final_weight") is None else float(d.get("final_weight"))),
            raw_weight=(None if d.get("raw_weight") is None else float(d.get("raw_weight"))),
            timestamp=(d.get("weightTimestamp") or d.get("weight_timestamp")),
            date=(d.get("weightDate") or d.get("weight_date")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        val = self.final_weight if self.final_weight is not None else self.raw_weight
        return f"{self.symbol}:{val}"


@dataclass
class PortfolioTrade:
    id: str
    symbol: str
    side: str
    qty: Optional[float]
    planned_price: Optional[float]
    planned_at: Optional[str]
    status: str
    reasoning: Optional[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortfolioTrade":
        return PortfolioTrade(
            id=str(d.get("id") or ""),
            symbol=str(d.get("symbol") or ""),
            side=str(d.get("side") or "").upper(),
            qty=(None if d.get("qty") is None else float(d.get("qty"))),
            planned_price=(None if d.get("planned_price") is None else float(d.get("planned_price"))),
            planned_at=(d.get("planned_at") or None),
            status=str(d.get("status") or "").upper(),
            reasoning=d.get("reasoning"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        return f"{self.symbol} {self.side} {self.status} x{self.qty}"


@dataclass
class PortfolioMember:
    strategy_id: str
    effective_weight: Optional[float]
    name: Optional[str] = None
    visibility: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PortfolioMember":
        return PortfolioMember(
            strategy_id=str(d.get("strategy_id") or ""),
            effective_weight=(None if d.get("effective_weight") is None else float(d.get("effective_weight"))),
            name=d.get("name"),
            visibility=d.get("visibility"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        return f"{self.name or self.strategy_id}:{self.effective_weight}"


@dataclass
class EquityPoint:
    date: str
    value: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EquityPoint":
        return EquityPoint(
            date=str(d.get("date") or ""),
            value=float(d.get("value") or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        return f"{self.date} {self.value}"


@dataclass
class LiveStatus:
    latest_run_id: Optional[str]
    timeframe: Optional[str]
    next_eta_minutes: Optional[float]
    counts: Dict[str, int]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LiveStatus":
        lr = (d or {}).get("latestRun") or {}
        return LiveStatus(
            latest_run_id=(lr.get("id") if isinstance(lr, dict) else None),
            timeframe=(lr.get("timeframe") if isinstance(lr, dict) else None),
            next_eta_minutes=(None if d.get("nextEtaMinutes") is None else float(d.get("nextEtaMinutes"))),
            counts={k: int(v) for k, v in ((d.get("counts") or {}).items())},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        return f"runId={self.latest_run_id} tf={self.timeframe} nextEtaMin={self.next_eta_minutes} counts={self.counts}"


@dataclass
class LiveLatest:
    run_id: Optional[str]
    timeframe: Optional[str]
    signals: List[PortfolioSignal]
    weights: List[PortfolioWeight]
    fresh: Optional[bool]
    recomputed: Optional[bool]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LiveLatest":
        results = (d or {}).get("results") or {}
        run_id = d.get("runId") or None
        tf = d.get("timeframe") or None
        sigs = parse_signals(results)
        wts = parse_weights(results)
        return LiveLatest(
            run_id=run_id,
            timeframe=tf,
            signals=sigs,
            weights=wts,
            fresh=d.get("fresh"),
            recomputed=d.get("recomputed"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timeframe": self.timeframe,
            "signals": [s.to_dict() for s in self.signals],
            "weights": [w.to_dict() for w in self.weights],
            "fresh": self.fresh,
            "recomputed": self.recomputed,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def log(self) -> str:
        return (
            f"runId={self.run_id} tf={self.timeframe} "
            f"signals={len(self.signals)} weights={len(self.weights)} fresh={self.fresh} recomputed={self.recomputed}"
        )


def parse_signals(payload: Dict[str, Any]) -> List[PortfolioSignal]:
    rows = (payload or {}).get("signals", []) if isinstance(payload, dict) else []
    return [PortfolioSignal.from_dict(r) for r in rows]


def parse_weights(payload: Dict[str, Any]) -> List[PortfolioWeight]:
    rows = (payload or {}).get("weights", []) if isinstance(payload, dict) else []
    return [PortfolioWeight.from_dict(r) for r in rows]


def parse_trades(payload: Dict[str, Any]) -> List[PortfolioTrade]:
    rows = (payload or {}).get("trades", []) if isinstance(payload, dict) else []
    return [PortfolioTrade.from_dict(r) for r in rows]


def parse_members(payload: Dict[str, Any]) -> List[PortfolioMember]:
    rows = (payload or {}).get("members", []) if isinstance(payload, dict) else []
    return [PortfolioMember.from_dict(r) for r in rows]


def parse_equity(payload: Dict[str, Any]) -> List[EquityPoint]:
    rows = (payload or {}).get("series", []) if isinstance(payload, dict) else []
    return [EquityPoint.from_dict(r) for r in rows]


def parse_live_latest(payload: Dict[str, Any]) -> LiveLatest:
    return LiveLatest.from_dict(payload or {})


