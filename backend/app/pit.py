"""Point-in-time 维度层: 回测防未来函数 (look-ahead bias) 的唯一读写口径。

慢变维度 (标的名称/ST、总股本/流通股本、上市状态、指数成分) 以**观察记录**
累积存储, 每条记录带两个时间戳:

- ``effective_at``: 状态在业务上生效的日期;
- ``known_at``: 本地实际可知 (观察/公告入库) 的日期。

回测对交易日 ``d`` 的任何维度查询统一走 as-of 口径:
``known_at <= d`` 的最后一条已知状态, 之后发布或修订的值物理上不可见,
因此同一历史日期读不到未来信息; 修订记录逐条保留, 旧值可复原。

存储布局 (用户数据目录, 不入库 Git):

- ``pit/instruments_history/part.parquet``: 标的维度每日观察累积;
  退市标的不再出现在当日快照时, 追加一条 ``status='delisted'`` 的观察,
  因此退市股与历史名称/股本/ST 轨迹不会消失。
- ``pit/index_constituents/part.parquet``: 指数成分完整快照观察;
  as-of 取 ``known_at <= d`` 的最近一份快照, 历史成分可复原。

财务指标/历史股本本身已按公告日门控 (见 backtest/fundamentals.py、
share_capital.py), 本模块只为它们产出审计证据, 不重复存储。
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

PIT_DIRNAME = "pit"
INSTRUMENTS_HISTORY_DIR = "instruments_history"
CONSTITUENTS_DIR = "index_constituents"

# 名称中这些 token 视为风险警示 (与 backtest/matrix.py 的 exclude_st 口径一致)。
_ST_NAME_TOKENS = ("ST", "*ST", "退")

_INSTRUMENT_VALUE_COLUMNS = (
    "name",
    "total_shares",
    "float_shares",
    "listing_date",
    "exchange",
    "status",
)
_INSTRUMENT_SCHEMA = {
    "symbol": pl.Utf8,
    "name": pl.Utf8,
    "total_shares": pl.Float64,
    "float_shares": pl.Float64,
    "listing_date": pl.Date,
    "exchange": pl.Utf8,
    "status": pl.Utf8,
    "effective_at": pl.Date,
    "known_at": pl.Date,
}
_CONSTITUENT_SCHEMA = {
    "index_symbol": pl.Utf8,
    "symbol": pl.Utf8,
    "weight": pl.Float64,
    "effective_at": pl.Date,
    "known_at": pl.Date,
}

# 注入 MarketDataMatrix.fields 的时变维度字段名。
PIT_IS_ST = "pit_is_st"
PIT_TOTAL_SHARES = "pit_total_shares"
PIT_FLOAT_SHARES = "pit_float_shares"
PIT_LISTED = "pit_listed"
PIT_MEMBER = "pit_member"
PIT_FIELD_NAMES = frozenset(
    {PIT_IS_ST, PIT_TOTAL_SHARES, PIT_FLOAT_SHARES, PIT_LISTED, PIT_MEMBER}
)

AUDIT_KEY = "pit_audit"


# ================================================================
# 写入侧: 观察记录
# ====================================================


def _pit_path(data_dir: Path, *parts: str) -> Path:
    return Path(data_dir) / PIT_DIRNAME / Path(*parts) / "part.parquet"


def _to_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _coerce_history(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    columns = {name: frame[name] for name in schema if name in frame.columns}
    out = pl.DataFrame(columns, schema={name: schema[name] for name in columns})
    missing = [name for name in schema if name not in out.columns]
    if missing:
        out = out.with_columns([
            pl.lit(None, dtype=schema[name]).alias(name) for name in missing
        ])
    return out.select(sorted(schema))


def _write_history(frame: pl.DataFrame, path: Path) -> None:
    from app.services.fs_utils import atomic_write_parquet

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(frame, path)


def record_instrument_observations(
    data_dir: Path,
    snapshot: pl.DataFrame,
    observed_at: date,
) -> int:
    """把当日 instruments 快照累积为 PIT 观察; 缺席的历史标的标记为退市。

    返回净新增的观察行数。内容与上一条观察完全相同的快照不重复写入
    (该状态的可知时间保守取首次观察日), 状态翻转 (含退市) 必留痕。
    """
    observed = _to_date(observed_at) or date.today()
    path = _pit_path(data_dir, INSTRUMENTS_HISTORY_DIR)
    history = load_instrument_history(data_dir)

    rows: list[dict[str, Any]] = []
    if snapshot is not None and not snapshot.is_empty() and "symbol" in snapshot.columns:
        for item in snapshot.iter_rows(named=True):
            symbol = item.get("symbol")
            if not symbol:
                continue
            rows.append({
                "symbol": str(symbol),
                "name": item.get("name"),
                "total_shares": item.get("total_shares"),
                "float_shares": item.get("float_shares"),
                "listing_date": _to_date(item.get("listing_date")),
                "exchange": item.get("exchange"),
                "status": "active",
                "effective_at": observed,
                "known_at": observed,
            })

    # 退市标记: 历史中最后状态为 active、当日快照缺席的标的各留一条观察。
    snapshot_symbols = {row["symbol"] for row in rows}
    if not history.is_empty():
        last_idx = history.group_by("symbol", maintain_order=True).agg(
            pl.all().sort_by("known_at").last()
        )
        for item in last_idx.iter_rows(named=True):
            if item.get("status") == "active" and item["symbol"] not in snapshot_symbols:
                rows.append({
                    "symbol": item["symbol"],
                    "name": item.get("name"),
                    "total_shares": item.get("total_shares"),
                    "float_shares": item.get("float_shares"),
                    "listing_date": item.get("listing_date"),
                    "exchange": item.get("exchange"),
                    "status": "delisted",
                    "effective_at": observed,
                    "known_at": observed,
                })

    if not rows:
        return 0

    incoming = _coerce_history(pl.DataFrame(rows), _INSTRUMENT_SCHEMA)
    merged = pl.concat([history, incoming], how="diagonal_relaxed")
    deduped = _dedupe_unchanged_observations(merged)
    old_keys = set(
        zip(
            history["symbol"].to_list() if not history.is_empty() else [],
            history["known_at"].to_list() if not history.is_empty() else [],
            strict=False,
        )
    )
    new_keys = set(zip(deduped["symbol"].to_list(), deduped["known_at"].to_list(), strict=False))
    added = len(new_keys - old_keys)
    if added or history.is_empty():
        _write_history(deduped, path)
    return added


def _dedupe_unchanged_observations(history: pl.DataFrame) -> pl.DataFrame:
    """同一 symbol 内容不变的相邻观察只保留最早一条; 任何字段翻转都留痕。"""
    if history.is_empty():
        return history
    ordered = history.sort(["symbol", "known_at", "effective_at"])
    value_cols = [c for c in _INSTRUMENT_VALUE_COLUMNS]
    lagged = [pl.col(c).shift(1).over("symbol").alias(f"_prev_{c}") for c in value_cols]
    marked = ordered.with_columns(lagged)
    first_of_symbol = pl.int_range(pl.len()).over("symbol") == 0
    changed_expr = [
        (pl.col(c).is_not_null() | pl.col(f"_prev_{c}").is_not_null())
        & (pl.col(c) != pl.col(f"_prev_{c}"))
        for c in value_cols
    ]
    keep = first_of_symbol
    for expr in changed_expr:
        keep = keep | expr.fill_null(True)
    return (
        marked.filter(keep)
        .drop([f"_prev_{c}" for c in value_cols])
        .sort(["symbol", "known_at"])
    )


def load_instrument_history(data_dir: Path) -> pl.DataFrame:
    """读取标的 PIT 观察历史; 不存在/损坏时返回空表 (带 schema)。"""
    path = _pit_path(data_dir, INSTRUMENTS_HISTORY_DIR)
    if not path.exists():
        return pl.DataFrame(schema=_INSTRUMENT_SCHEMA)
    try:
        frame = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 instruments PIT 历史失败: %s", exc)
        return pl.DataFrame(schema=_INSTRUMENT_SCHEMA)
    return _coerce_history(frame, _INSTRUMENT_SCHEMA).sort(["symbol", "known_at"])


def record_constituent_observations(
    data_dir: Path,
    index_symbol: str,
    members: Iterable[str] | pl.DataFrame,
    observed_at: date,
) -> int:
    """记录某指数当日完整成分快照; 成分集合与上一份快照相同则跳过。

    ``members`` 为成分 symbol 可迭代对象, 或含 ``symbol`` 列的 DataFrame
    (可选 ``weight`` 列)。
    """
    observed = _to_date(observed_at) or date.today()
    index_symbol = str(index_symbol)
    history = load_constituent_history(data_dir, index_symbol)

    if isinstance(members, pl.DataFrame):
        frame = members
        symbols = (
            frame["symbol"].cast(pl.Utf8).drop_nulls().unique().to_list()
            if "symbol" in frame.columns else []
        )
        weights: dict[str, Any] = {}
        if "weight" in frame.columns:
            for item in frame.select(["symbol", "weight"]).iter_rows(named=True):
                if item.get("symbol") is not None:
                    weights[str(item["symbol"])] = item.get("weight")
    else:
        symbols = sorted({str(s) for s in members if s})
        weights = {}

    previous: set[str] = set()
    if not history.is_empty():
        latest_known = history["known_at"].max()
        previous = set(
            history.filter(pl.col("known_at") == latest_known)["symbol"].to_list()
        )
    if set(symbols) == previous and previous:
        return 0
    if not symbols:
        return 0

    rows = [
        {
            "index_symbol": index_symbol,
            "symbol": symbol,
            "weight": weights.get(symbol),
            "effective_at": observed,
            "known_at": observed,
        }
        for symbol in symbols
    ]
    incoming = _coerce_history(pl.DataFrame(rows), _CONSTITUENT_SCHEMA)
    merged = pl.concat([history, incoming], how="diagonal_relaxed")
    _write_history(merged, _pit_path(data_dir, CONSTITUENTS_DIR))
    return len(rows)


def load_constituent_history(
    data_dir: Path,
    index_symbol: str | None = None,
) -> pl.DataFrame:
    """读取指数成分 PIT 观察; ``index_symbol`` 非空时只取该指数。"""
    path = _pit_path(data_dir, CONSTITUENTS_DIR)
    if not path.exists():
        return pl.DataFrame(schema=_CONSTITUENT_SCHEMA)
    try:
        frame = pl.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取指数成分 PIT 历史失败: %s", exc)
        return pl.DataFrame(schema=_CONSTITUENT_SCHEMA)
    frame = _coerce_history(frame, _CONSTITUENT_SCHEMA)
    if index_symbol is not None:
        frame = frame.filter(pl.col("index_symbol") == str(index_symbol))
    return frame.sort(["index_symbol", "known_at", "symbol"])


def constituents_as_of(
    history: pl.DataFrame,
    index_symbol: str,
    as_of: date,
) -> set[str]:
    """复原 ``as_of`` 当日可知的最新成分集合 (known_at <= as_of)。"""
    if history.is_empty():
        return set()
    target = _to_date(as_of)
    frame = history.filter(
        (pl.col("index_symbol") == str(index_symbol))
        & (pl.col("known_at").is_not_null())
        & (pl.col("known_at") <= target)
    )
    if frame.is_empty():
        return set()
    latest_known = frame["known_at"].max()
    return set(
        frame.filter(pl.col("known_at") == latest_known)["symbol"].drop_nulls().to_list()
    )


# ================================================================
# 读取侧: as-of 矩阵
# ================================================================


def _is_st_name(name: Any) -> bool:
    if not name:
        return False
    upper = str(name).upper()
    return any(token in upper for token in _ST_NAME_TOKENS)


def _session_dates(market: Any) -> tuple[np.ndarray, np.ndarray]:
    """返回 (唯一 session 归属交易日 datetime64[D], 每根 bar 的 session 序号)。

    不假设 session_id 从 0 连续编号; 序号为 np.unique 给出的稠密下标。
    """
    labels = market.timestamp_labels
    session_values = np.asarray(market.session_ids)
    unique_sessions, inverse = np.unique(session_values, return_inverse=True)
    dates = np.empty(len(unique_sessions), dtype="datetime64[D]")
    for ordinal, sid in enumerate(unique_sessions):
        first_index = int(np.flatnonzero(session_values == sid)[0])
        dates[ordinal] = np.datetime64(str(labels[first_index])[:10], "D")
    return dates, inverse


def build_pit_field_matrices(
    market: Any,
    history: pl.DataFrame,
    *,
    member_symbols_by_date: dict[date, set[str]] | None = None,
) -> dict[str, np.ndarray]:
    """把 PIT 维度历史物化为 T×N 矩阵 (与 MarketDataMatrix 同轴)。

    纯 as-of 构造: 对每个交易日 d 只取 ``known_at <= d`` 的最后观察,
    future 观察物理上不进入索引 (searchsorted side='right'),
    因此未来发布/修订的值不会在任何历史日期可见。
    """
    shape = market.shape
    session_dates, session_ordinal = _session_dates(market) if shape[0] else (
        np.empty(0, dtype="datetime64[D]"), np.empty(0, dtype=np.intp)
    )
    n_sessions = len(session_dates)
    st_sessions = np.zeros((n_sessions, shape[1]), dtype=bool)
    listed_sessions = np.ones((n_sessions, shape[1]), dtype=bool)
    total_sessions = np.full((n_sessions, shape[1]), np.nan, dtype=np.float64)
    float_sessions = np.full((n_sessions, shape[1]), np.nan, dtype=np.float64)

    if history is not None and not history.is_empty():
        asset_index = {
            symbol: index for index, symbol in enumerate(market.symbols)
        }
        known = history.with_columns(
            pl.col("known_at").cast(pl.Date)
        ).sort(["symbol", "known_at"])
        for symbol, group in known.group_by("symbol", maintain_order=True):
            sym = symbol[0] if isinstance(symbol, tuple) else symbol
            column = asset_index.get(str(sym))
            if column is None:
                continue
            known_dates = np.array(
                [d.isoformat() for d in group["known_at"].to_list() if d is not None],
                dtype="datetime64[D]",
            )
            if known_dates.size == 0:
                continue
            cutoff = np.searchsorted(known_dates, session_dates, side="right")
            names = group["name"].to_list()
            statuses = group["status"].to_list()
            totals = group["total_shares"].to_list()
            floats = group["float_shares"].to_list()
            listings = [
                np.datetime64(d.isoformat(), "D") if d is not None else None
                for d in group["listing_date"].to_list()
            ]
            last_total: float | None = None
            last_float: float | None = None
            for row, d in enumerate(session_dates):
                idx = int(cutoff[row]) - 1
                if idx < 0:
                    continue
                st_sessions[row, column] = _is_st_name(names[idx])
                listing = listings[idx]
                listed_sessions[row, column] = (
                    listing is not None and d >= listing
                    and statuses[idx] != "delisted"
                )
                # 股本允许跨观察前向填充非空值 (修订/补发空值不回退)。
                for back in range(idx, -1, -1):
                    value = totals[back]
                    if value is not None:
                        last_total = float(value)
                        break
                for back in range(idx, -1, -1):
                    value = floats[back]
                    if value is not None:
                        last_float = float(value)
                        break
                if last_total is not None:
                    total_sessions[row, column] = last_total
                if last_float is not None:
                    float_sessions[row, column] = last_float

    member_sessions = np.ones((n_sessions, shape[1]), dtype=bool)
    if member_symbols_by_date:
        member_sessions.fill(False)
        asset_index = {s: i for i, s in enumerate(market.symbols)}
        # as-of: 非成分观察日沿用最近一份已知快照 (known_at <= d), 不做精确日匹配。
        obs_pairs = sorted(member_symbols_by_date.items())
        obs_dates = np.array(
            [d.isoformat() for d, _ in obs_pairs], dtype="datetime64[D]"
        )
        cutoff = np.searchsorted(obs_dates, session_dates, side="right") - 1
        for row, idx in enumerate(cutoff):
            if idx < 0:
                continue
            ids = [asset_index[s] for s in obs_pairs[int(idx)][1] if s in asset_index]
            if ids:
                member_sessions[row, ids] = True

    gather = session_ordinal
    arrays = {
        PIT_IS_ST: st_sessions[gather],
        PIT_TOTAL_SHARES: total_sessions[gather],
        PIT_FLOAT_SHARES: float_sessions[gather],
        PIT_LISTED: listed_sessions[gather],
        PIT_MEMBER: member_sessions[gather],
    }
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def replace_matrix(market: Any, **changes: Any) -> Any:
    """复制矩阵对象并覆盖字段 (兼容 MarketDataMatrix 与鸭子类型测试替身)。"""
    import dataclasses

    if dataclasses.is_dataclass(market) and not isinstance(market, type):
        return dataclasses.replace(market, **changes)
    from types import SimpleNamespace

    return SimpleNamespace(**vars(market), **changes)


def attach_pit_fields_to_market(
    market: Any,
    history: pl.DataFrame | None,
    *,
    member_symbols_by_date: dict[date, set[str]] | None = None,
) -> Any:
    """返回附加 PIT 字段的 MarketDataMatrix 副本 (字段只读)。
    """
    if (history is None or history.is_empty()) and not member_symbols_by_date:
        return market
    extra = build_pit_field_matrices(
        market, history, member_symbols_by_date=member_symbols_by_date
    )
    merged = {**dict(market.fields), **extra}
    for array in extra.values():
        array.setflags(write=False)
    return replace_matrix(market, fields=MappingProxyType(merged))


def attach_pit_columns_panel(
    panel: pl.DataFrame,
    history: pl.DataFrame | None,
) -> pl.DataFrame:
    """为 polars 面板按 (symbol, date) as-of 附加 PIT 维度列。

    供矩阵化之前的面板准备路径使用; 同样严格按 ``known_at <= date`` 回看。
    """
    if history is None or history.is_empty() or panel.is_empty():
        return panel
    if not {"symbol", "date"} <= set(panel.columns):
        return panel

    # 统一按 Date 做 as-of (面板 date 列可能是 Datetime)。
    panel_date_is_date = panel.schema["date"] == pl.Date
    dated_panel = panel if panel_date_is_date else panel.with_columns(
        pl.col("date").cast(pl.Date).alias("_pit_panel_date")
    )
    panel_date_col = "date" if panel_date_is_date else "_pit_panel_date"

    right = history.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("known_at").alias("_pit_known_at"),
        pl.col("name").alias("_pit_name"),
        pl.col("total_shares").cast(pl.Float64).alias("_pit_total_shares"),
        pl.col("float_shares").cast(pl.Float64).alias("_pit_float_shares"),
        pl.col("listing_date").alias("_pit_listing_date"),
    ).sort(["symbol", "_pit_known_at"])

    ordered = dated_panel.with_row_index("_pit_row_order").sort(["symbol", panel_date_col])
    joined = ordered.join_asof(
        right,
        left_on=panel_date_col,
        right_on="_pit_known_at",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    from app.price_limits import polars_is_risk_warning_name

    st_expr = polars_is_risk_warning_name(pl.col("_pit_name"))
    joined = joined.with_columns(
        st_expr.alias(PIT_IS_ST),
        pl.col("_pit_total_shares").cast(pl.Float64).alias(PIT_TOTAL_SHARES),
        pl.col("_pit_float_shares").cast(pl.Float64).alias(PIT_FLOAT_SHARES),
        (
            pl.col("_pit_listing_date").is_not_null()
            & (pl.col(panel_date_col) >= pl.col("_pit_listing_date"))
        ).alias(PIT_LISTED),
    )
    # 运行时 look-ahead lint: as-of join 不得让任何行看到未来观察。
    raise_if_leakage(guard_panel_asof(
        joined,
        date_column=panel_date_col,
        time_column="_pit_known_at",
        fields=[PIT_IS_ST, PIT_TOTAL_SHARES, PIT_FLOAT_SHARES],
        source="pit/instruments_history join_asof",
    ))
    return _drop_pit_helpers(joined)


def _drop_pit_helpers(joined: pl.DataFrame) -> pl.DataFrame:
    helpers = [
        c for c in (
            "_pit_row_order", "_pit_known_at", "_pit_name",
            "_pit_total_shares", "_pit_float_shares", "_pit_listing_date",
            "_pit_panel_date",
        ) if c in joined.columns
    ]
    if not helpers:
        return joined
    return joined.sort("_pit_row_order").drop(helpers) if "_pit_row_order" in helpers else joined.drop(helpers)


# ================================================================
# Universe 定义
# ================================================================


@dataclass(frozen=True)
class UniverseResolution:
    symbols: tuple[str, ...] | None
    definition: MappingProxyType[str, Any]


def resolve_matrix_universe(
    *,
    explicit_symbols: list[str] | tuple[str, ...] | None,
    index_symbol: str | None,
    data_dir: Path | None,
    start: date,
    end: date,
    asset_type: str = "stock",
) -> UniverseResolution:
    """解析矩阵回测的股票池。

    - 显式 symbols: ``explicit`` 模式;
    - ``index_symbol``: ``index`` 模式, 窗口内各 as-of 成分的并集,
      逐日归属由 ``pit_member`` 矩阵门控; 无成分历史时 fail-closed;
    - 均为空: ``all_listed`` 模式, 轴取本地行情中出现过的全部标的
      (含已退市, 由 PIT 上市/状态矩阵逐日门控)。
    """
    definition: dict[str, Any] = {
        "asset_type": asset_type,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "as_of_rule": "known_at <= trade_date; fundamentals effective_at=announce_date+1",
        "delisted_included": True,
        "revisions": "point_in_time_rows_kept",
    }
    if asset_type != "stock":
        definition["mode"] = "all_assets"
        definition["delisted_included"] = False
        return UniverseResolution(tuple(explicit_symbols) if explicit_symbols else None,
                                 MappingProxyType(definition))

    if explicit_symbols:
        symbols = tuple(sorted({str(s) for s in explicit_symbols if s}))
        definition.update({"mode": "explicit", "symbol_count": len(symbols)})
        return UniverseResolution(symbols, MappingProxyType(definition))

    if index_symbol:
        history = load_constituent_history(data_dir, index_symbol) if data_dir else pl.DataFrame(
            schema=_CONSTITUENT_SCHEMA
        )
        if history.is_empty():
            raise ValueError(
                f"指数 {index_symbol} 无 point-in-time 成分历史, 无法按历史成分复原 universe; "
                "请先同步指数成分, 或改用显式 symbols / 全市场模式"
            )
        members_by_date = _member_sets_by_observation(history, start, end)
        union: set[str] = set()
        for members in members_by_date.values():
            union.update(members)
        if not union:
            union = constituents_as_of(history, index_symbol, end)
        if not union:
            raise ValueError(f"指数 {index_symbol} 成分历史不覆盖回测窗口 {start}~{end}")
        definition.update({
            "mode": "index",
            "index_symbol": str(index_symbol),
            "symbol_count": len(union),
            "observation_dates": len(members_by_date),
            "source": str(_pit_path(data_dir, CONSTITUENTS_DIR)),
        })
        return UniverseResolution(tuple(sorted(union)), MappingProxyType(definition))

    definition.update({
        "mode": "all_listed",
        "symbol_count": None,
        "note": "axis includes delisted symbols retained in local kline partitions; "
                "per-date listing/ST gates applied via pit_* matrices",
        "source": str(_pit_path(data_dir, INSTRUMENTS_HISTORY_DIR)) if data_dir else None,
    })
    return UniverseResolution(None, MappingProxyType(definition))


def member_sets_by_window(
    history: pl.DataFrame,
    index_symbol: str | None,
    start: date,
    end: date,
) -> dict[date, set[str]]:
    """窗口内每个成分观察日对应的完整成分集合 (供 pit_member 逐日门控)。"""
    if history is None or history.is_empty() or not index_symbol:
        return {}
    return _member_sets_by_observation(
        history.filter(pl.col("index_symbol") == str(index_symbol)), start, end
    )


def _member_sets_by_observation(
    history: pl.DataFrame,
    start: date,
    end: date,
) -> dict[date, set[str]]:
    if history.is_empty():
        return {}
    frame = history.filter(
        pl.col("known_at").is_not_null()
        & (pl.col("known_at") >= _to_date(start))
        & (pl.col("known_at") <= _to_date(end))
    )
    out: dict[date, set[str]] = {}
    for known, group in frame.group_by("known_at"):
        known_day = known[0] if isinstance(known, tuple) else known
        if known_day is not None:
            out[known_day] = set(group["symbol"].drop_nulls().to_list())
    return dict(sorted(out.items()))


# ================================================================
# Leakage 审计
# ================================================================


@dataclass(frozen=True)
class LeakageViolation:
    field: str
    source: str
    symbol: str | None
    as_of: str
    known_at: str

    def evidence(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "source": self.source,
            "symbol": self.symbol,
            "as_of": self.as_of,
            "known_at": self.known_at,
        }


class FutureLeakageError(RuntimeError):
    """历史日期读取到了之后才可知的维度值; 任务必须失败。"""

    def __init__(self, violations_or_message: list[LeakageViolation] | str) -> None:
        if isinstance(violations_or_message, str):
            self.violations: list[LeakageViolation] = []
            super().__init__(violations_or_message)
            return
        self.violations = violations_or_message
        lines = ["检测到 future-known 维度字段进入历史回测, 任务终止:"]
        for item in self.violations[:20]:
            lines.append(
                f"- field={item.field} symbol={item.symbol} "
                f"as_of={item.as_of} 但 known_at={item.known_at} (source={item.source})"
            )
        if len(self.violations) > 20:
            lines.append(f"… 其余 {len(self.violations) - 20} 条见 leakage_report")
        super().__init__("\n".join(lines))


def guard_frame_asof(
    frame: pl.DataFrame,
    as_of: date | str,
    *,
    fields: Iterable[str],
    time_column: str = "known_at",
    source: str = "pit",
    symbol_column: str = "symbol",
    limit: int = 50,
) -> list[LeakageViolation]:
    """lint 原语: 断言帧内没有 as_of 之后才可知的非空字段值。

    任何接入回测的维度帧都应先经过本守卫 (或等价的 as-of join);
    返回违例证据列表, 非空时调用方应抛 ``FutureLeakageError``。
    """
    if frame is None or frame.is_empty() or time_column not in frame.columns:
        return []
    target = _to_date(as_of)
    if target is None:
        return []
    time_expr = pl.col(time_column)
    if frame.schema[time_column] == pl.Utf8:
        time_expr = pl.col(time_column).str.slice(0, 10).str.to_date(strict=False)
    bad = frame.filter(time_expr.is_not_null() & (time_expr > target))
    if bad.is_empty():
        return []
    violations: list[LeakageViolation] = []
    symbols = bad[symbol_column].to_list() if symbol_column in bad.columns else [None] * bad.height
    times = bad.select(time_expr.alias("_t"))["_t"].to_list()
    for field_name in fields:
        if field_name not in bad.columns:
            continue
        values = bad[field_name].to_list()
        for row, value in enumerate(values):
            if value is None:
                continue
            violations.append(LeakageViolation(
                field=str(field_name),
                source=source,
                symbol=str(symbols[row]) if symbols[row] is not None else None,
                as_of=target.isoformat(),
                known_at=times[row].isoformat() if times[row] is not None else "unknown",
            ))
            if len(violations) >= limit:
                return violations
    return violations


def raise_if_leakage(violations: list[LeakageViolation]) -> None:
    if violations:
        raise FutureLeakageError(violations)


def guard_panel_asof(
    frame: pl.DataFrame,
    *,
    date_column: str = "date",
    time_column: str = "known_at",
    fields: Iterable[str],
    source: str = "pit_join_asof",
    symbol_column: str = "symbol",
    limit: int = 50,
) -> list[LeakageViolation]:
    """逐行 lint: 每行实际可见维度的 known_at 不得晚于该行交易日。

    接在 as-of join 之后做防御性不变量检查; join 语义被破坏 (如误用 forward
    join) 时会在此产生违例证据, 由调用方决定 fail-fast。
    """
    if (
        frame is None
        or frame.is_empty()
        or time_column not in frame.columns
        or date_column not in frame.columns
    ):
        return []
    known = pl.col(time_column)
    if frame.schema[time_column] == pl.Utf8:
        known = known.str.slice(0, 10).str.to_date(strict=False)
    bad = frame.filter(
        known.is_not_null() & (known > pl.col(date_column).cast(pl.Date))
    )
    if bad.is_empty():
        return []
    violations: list[LeakageViolation] = []
    for row in bad.iter_rows(named=True):
        known_day = row.get(time_column)
        as_of_day = _to_date(row.get(date_column))
        symbol = row.get(symbol_column)
        for field_name in fields:
            if field_name not in bad.columns or row.get(field_name) is None:
                continue
            violations.append(LeakageViolation(
                field=str(field_name),
                source=source,
                symbol=str(symbol) if symbol is not None else None,
                as_of=as_of_day.isoformat() if as_of_day else "unknown",
                known_at=str(_to_date(known_day)) if known_day else "unknown",
            ))
            if len(violations) >= limit:
                return violations
    return violations


def build_leakage_report(dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总各维度的 as-of 审计结果; violation 存在时整体状态为 violation。"""
    has_violation = any(d.get("status") == "violation" for d in dimensions)
    degraded = any(d.get("status") == "degraded" for d in dimensions)
    return {
        "status": "violation" if has_violation else ("degraded" if degraded else "ok"),
        "rule": "every dimension value visible at date d must have known_at <= d",
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dimensions": dimensions,
    }


def instrument_dimension_audit(
    *,
    history: pl.DataFrame | None,
    pit_used: bool,
    snapshot: pl.DataFrame | None,
    start: date,
    fields_used: Iterable[str] = ("name", "total_shares", "float_shares", "listing_date"),
) -> dict[str, Any]:
    """标的维度审计: 严格 as-of 或记录快照降级证据。"""
    if pit_used and history is not None and not history.is_empty():
        future = history.filter(pl.col("known_at") > start)
        return {
            "name": "instruments",
            "status": "ok",
            "mode": "pit_asof",
            "table": f"{PIT_DIRNAME}/{INSTRUMENTS_HISTORY_DIR}/part.parquet",
            "fields": list(fields_used),
            "observations": history.height,
            "future_observations_excluded": future.height,
        }
    evidence: list[dict[str, Any]] = []
    status = "unavailable"
    if snapshot is not None and not snapshot.is_empty() and "as_of" in snapshot.columns:
        latest = snapshot["as_of"].drop_nulls().max()
        latest_day = _to_date(latest)
        if latest_day is not None and latest_day > start:
            status = "degraded"
            sample = snapshot.select(
                pl.col("symbol"), pl.col("as_of")
            ).head(5)
            for row in sample.iter_rows(named=True):
                evidence.append({
                    "field": "instruments_snapshot",
                    "source": "instruments/instruments.parquet",
                    "symbol": row.get("symbol"),
                    "as_of": start.isoformat(),
                    "known_at": str(row["as_of"])[:10],
                })
    return {
        "name": "instruments",
        "status": status,
        "mode": "snapshot_fallback" if status == "degraded" else "unavailable",
        "fields": list(fields_used),
        "evidence": evidence,
        "warning": "PIT 标的历史缺失, 使用最新维表快照, 历史日期可能含未来可知信息"
        if status == "degraded" else "无 instruments 维表",
    }


def constituent_dimension_audit(
    *,
    history: pl.DataFrame | None,
    index_symbol: str | None,
) -> dict[str, Any]:
    if not index_symbol:
        return {
            "name": "index_constituents",
            "status": "ok",
            "mode": "not_used",
        }
    if history is None or history.is_empty():
        return {
            "name": "index_constituents",
            "status": "degraded",
            "mode": "unavailable",
            "index_symbol": index_symbol,
            "warning": "无指数成分历史, universe 无法按历史成分复原",
        }
    return {
        "name": "index_constituents",
        "status": "ok",
        "mode": "pit_asof",
        "index_symbol": index_symbol,
        "observations": history.filter(
            pl.col("index_symbol") == str(index_symbol)
        ).height,
    }


def announcement_dimension_audit(
    *,
    table: str,
    rows: int,
    gate: str,
    available: bool,
) -> dict[str, Any]:
    """财务指标/历史股本等已按公告日门控维度的审计留痕。"""
    return {
        "name": table,
        "status": "ok" if available else "unavailable",
        "mode": "announce_date_gated" if available else "missing",
        "gate": gate,
        "rows": rows,
    }
