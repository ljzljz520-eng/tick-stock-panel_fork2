"""Point-in-time 维度层 / 回测防未来函数测试。

覆盖验收契约:
- 同一历史日期读不到之后发布/修订的名称、ST、股本值;
- 退市标的保留且退市后被逐日门控, 修订旧值可复原;
- 指数成分按 as-of 快照复原, 无历史时 fail-closed;
- guard_frame_asof 给出字段+时间证据, 违例抛 FutureLeakageError;
- basic filter 的 ST/市值判定使用逐日 PIT 矩阵;
- 读路径接线 (engine/matrix) 不被绕过的静态回归守卫。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from app import pit
from app.backtest.matrix import (
    PIT_IS_ST,
    PIT_LISTED,
    PIT_MEMBER,
    PIT_TOTAL_SHARES,
    _build_basic_filter_mask_uncached,
    build_market_data_matrix,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]

D1 = date(2024, 1, 1)
D2 = date(2024, 1, 2)
D3 = date(2024, 1, 3)
D4 = date(2024, 1, 4)
D5 = date(2024, 1, 5)
D6 = date(2024, 1, 6)
AAA = "000001.SZ"
BBB = "600000.SH"
INDEX = "000300.SH"


def _snapshot(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["symbol", "name", "total_shares", "float_shares", "listing_date"],
        orient="row",
    ).with_columns([
        pl.lit("X").alias("exchange"),
        pl.lit("active").alias("status"),
    ])


def _panel(symbols: tuple[str, ...], days: int = 6, start: date = D1) -> pl.DataFrame:
    rows = []
    for offset in range(days):
        for asset_id, symbol in enumerate(symbols):
            close = 10.0 + asset_id
            rows.append({
                "symbol": symbol,
                "date": start + timedelta(days=offset),
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": 1000.0,
            })
    return pl.DataFrame(rows)


def _day_rows(market, day: date) -> list[int]:
    prefix = day.isoformat()
    return [
        i for i, label in enumerate(market.timestamp_labels)
        if str(label)[:10] == prefix
    ]


def _cell(market, array_name: str, day: date, symbol: str):
    column = list(market.symbols).index(symbol)
    values = market.fields[array_name]
    hits = [values[r, column] for r in _day_rows(market, day)]
    assert len(hits) == 1, f"{day} {symbol} 应当只有一根 bar, 实际 {len(hits)}"
    return hits[0]


# ================================================================
# 写入侧: 观察累积、去重、退市留痕
# ================================================================


def test_record_observations_dedupes_unchanged_and_marks_delisted(tmp_path):
    snap1 = _snapshot([
        (AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    assert pit.record_instrument_observations(tmp_path, snap1, D1) == 2
    # 完全相同的次日快照不重复写入。
    assert pit.record_instrument_observations(tmp_path, snap1, D2) == 0

    snap3 = _snapshot([(BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1))])
    added = pit.record_instrument_observations(tmp_path, snap3, D3)
    # BBB 无变化被去重; AAA 缺席 → 一条 delisted 观察。
    assert added == 1

    history = pit.load_instrument_history(tmp_path)
    aaa = history.filter(pl.col("symbol") == AAA).sort("known_at")
    assert aaa["status"].to_list() == ["active", "delisted"]
    assert aaa["known_at"].to_list() == [D1, D3]
    # 退市行保留退市前最后名称/股本, 历史轨迹可复原。
    last = aaa.row(-1, named=True)
    assert last["name"] == "甲公司"
    assert last["total_shares"] == 100.0


def test_revisions_leave_trace_and_old_value_recoverable(tmp_path):
    snap1 = _snapshot([(AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1))])
    snap2 = _snapshot([(AAA, "ST甲公司", 250.0, 90.0, date(2020, 1, 1))])
    pit.record_instrument_observations(tmp_path, snap1, D1)
    pit.record_instrument_observations(tmp_path, snap2, D4)

    history = pit.load_instrument_history(tmp_path)
    assert history.filter(pl.col("symbol") == AAA).height == 2

    market = pit.attach_pit_fields_to_market(
        build_market_data_matrix(_panel((AAA,))), history
    )
    # D2 只能看到 D1 的旧名称/旧股本; D4 才可见修订值。
    assert bool(_cell(market, PIT_IS_ST, D2, AAA)) is False
    assert _cell(market, PIT_TOTAL_SHARES, D2, AAA) == pytest.approx(100.0)
    assert bool(_cell(market, PIT_IS_ST, D4, AAA)) is True
    assert _cell(market, PIT_TOTAL_SHARES, D4, AAA) == pytest.approx(250.0)


# ================================================================
# as-of 读取: 未来注入不可见
# ================================================================


def test_future_known_name_shares_st_invisible_at_historical_date(tmp_path):
    snap1 = _snapshot([
        (AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    # D4 的快照把 AAA 改成 ST 并做大股本 (模拟"今天的维表")。
    snap4 = _snapshot([
        (AAA, "*ST甲公司", 9_999.0, 8_888.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    pit.record_instrument_observations(tmp_path, snap1, D1)
    pit.record_instrument_observations(tmp_path, snap4, D4)
    history = pit.load_instrument_history(tmp_path)

    market = pit.attach_pit_fields_to_market(
        build_market_data_matrix(_panel((AAA, BBB))), history
    )
    # 历史日期 D2: ST 与新股本物理上不可见。
    assert bool(_cell(market, PIT_IS_ST, D2, AAA)) is False
    assert _cell(market, PIT_TOTAL_SHARES, D2, AAA) == pytest.approx(100.0)
    assert _cell(market, "pit_float_shares", D2, AAA) == pytest.approx(80.0)
    # 发布日 D4 起可见 (known_at <= d 当条可见)。
    assert bool(_cell(market, PIT_IS_ST, D4, AAA)) is True
    assert _cell(market, PIT_TOTAL_SHARES, D4, AAA) == pytest.approx(9_999.0)
    # BBB 全天不受影响。
    assert bool(_cell(market, PIT_IS_ST, D6, BBB)) is False
    assert _cell(market, PIT_TOTAL_SHARES, D6, BBB) == pytest.approx(200.0)


def test_delisted_symbol_gated_after_delisting_observation(tmp_path):
    snap1 = _snapshot([
        (AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    snap3 = _snapshot([(BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1))])
    pit.record_instrument_observations(tmp_path, snap1, D1)
    pit.record_instrument_observations(tmp_path, snap3, D3)
    history = pit.load_instrument_history(tmp_path)

    market = pit.attach_pit_fields_to_market(
        build_market_data_matrix(_panel((AAA, BBB))), history
    )
    # 退市观察 D3 之前仍可交易; D3 起 pit_listed 门控关闭。
    assert bool(_cell(market, PIT_LISTED, D2, AAA)) is True
    assert bool(_cell(market, PIT_LISTED, D3, AAA)) is False
    assert bool(_cell(market, PIT_LISTED, D6, AAA)) is False
    assert bool(_cell(market, PIT_LISTED, D6, BBB)) is True


# ================================================================
# 指数成分: as-of 复原 + fail-closed + 逐日门控
# ================================================================


def test_constituents_asof_and_index_universe_fail_closed(tmp_path):
    pit.record_constituent_observations(tmp_path, INDEX, [AAA], D1)
    pit.record_constituent_observations(tmp_path, INDEX, [AAA, BBB], D3)
    history = pit.load_constituent_history(tmp_path, INDEX)

    assert pit.constituents_as_of(history, INDEX, D1) == {AAA}
    # 观察日之间沿用最近一份已知快照。
    assert pit.constituents_as_of(history, INDEX, D2) == {AAA}
    assert pit.constituents_as_of(history, INDEX, D3) == {AAA, BBB}
    assert pit.constituents_as_of(history, INDEX, date(2023, 12, 31)) == set()

    # 无成分历史 → fail-closed, 绝不静默退回全市场。
    with pytest.raises(ValueError, match="point-in-time 成分历史"):
        pit.resolve_matrix_universe(
            explicit_symbols=None,
            index_symbol="999999.SH",
            data_dir=tmp_path,
            start=D1,
            end=D6,
        )

    resolved = pit.resolve_matrix_universe(
        explicit_symbols=None,
        index_symbol=INDEX,
        data_dir=tmp_path,
        start=D1,
        end=D6,
    )
    assert set(resolved.symbols) == {AAA, BBB}
    assert resolved.definition["mode"] == "index"
    assert resolved.definition["delisted_included"] is True


def test_index_member_matrix_gates_per_date_with_forward_fill(tmp_path):
    pit.record_constituent_observations(tmp_path, INDEX, [AAA], D1)
    pit.record_constituent_observations(tmp_path, INDEX, [AAA, BBB], D4)
    history = pit.load_constituent_history(tmp_path, INDEX)
    members = pit.member_sets_by_window(history, INDEX, D1, D6)

    market = pit.attach_pit_fields_to_market(
        build_market_data_matrix(_panel((AAA, BBB))),
        pit.load_instrument_history(tmp_path),
        member_symbols_by_date=members,
    )
    assert bool(_cell(market, PIT_MEMBER, D1, AAA)) is True
    assert bool(_cell(market, PIT_MEMBER, D2, AAA)) is True  # 非观察日前向填充
    assert bool(_cell(market, PIT_MEMBER, D3, BBB)) is False
    assert bool(_cell(market, PIT_MEMBER, D4, BBB)) is True
    assert bool(_cell(market, PIT_MEMBER, D6, BBB)) is True


# ================================================================
# look-ahead lint: 证据 + 任务失败
# ================================================================


def test_guard_frame_asof_collects_evidence_and_raises():
    frame = pl.DataFrame([
        {"symbol": AAA, "known_at": D2, "total_shares": 100.0},
        {"symbol": BBB, "known_at": D5, "total_shares": 200.0},  # 未来修订
        {"symbol": AAA, "known_at": D6, "total_shares": None},   # 空值不算泄露
    ])
    violations = pit.guard_frame_asof(
        frame, D3, fields=["total_shares"], source="unit-test"
    )
    assert len(violations) == 1
    evidence = violations[0].evidence()
    assert evidence["field"] == "total_shares"
    assert evidence["symbol"] == BBB
    assert evidence["as_of"] == D3.isoformat()
    assert evidence["known_at"] == D5.isoformat()

    with pytest.raises(pit.FutureLeakageError) as excinfo:
        pit.raise_if_leakage(violations)
    message = str(excinfo.value)
    assert "total_shares" in message
    assert D5.isoformat() in message
    assert D3.isoformat() in message


def test_guard_frame_asof_passes_clean_frame():
    frame = pl.DataFrame([
        {"symbol": AAA, "known_at": D1, "name": "甲"},
        {"symbol": BBB, "known_at": D2, "name": "乙"},
    ])
    assert pit.guard_frame_asof(frame, D3, fields=["name"]) == []


def test_guard_panel_asof_flags_rowwise_future_join():
    # 第二行模拟被错误的 forward join 注入: D2 行读到了 D5 才可知的股本。
    panel = pl.DataFrame([
        {"symbol": AAA, "date": D1, "known_at": D1, PIT_TOTAL_SHARES: 100.0},
        {"symbol": AAA, "date": D2, "known_at": D5, PIT_TOTAL_SHARES: 999.0},
        {"symbol": BBB, "date": D6, "known_at": D5, PIT_TOTAL_SHARES: 200.0},
    ])
    violations = pit.guard_panel_asof(panel, fields=[PIT_TOTAL_SHARES])
    assert len(violations) == 1
    evidence = violations[0].evidence()
    assert evidence["symbol"] == AAA
    assert evidence["as_of"] == D2.isoformat()
    assert evidence["known_at"] == D5.isoformat()
    with pytest.raises(pit.FutureLeakageError):
        pit.raise_if_leakage(violations)

    # attach_pit_columns_panel 自身是 backward as-of, lint 不应误报。
    clean = pl.DataFrame([
        {"symbol": AAA, "date": D1, PIT_TOTAL_SHARES: 100.0},
        {"symbol": AAA, "date": D2, PIT_TOTAL_SHARES: 100.0},
    ])
    assert pit.guard_panel_asof(clean, fields=[PIT_TOTAL_SHARES]) == []


def test_leakage_report_status_precedence(tmp_path):
    history = pit.load_instrument_history(tmp_path)
    degraded = pit.instrument_dimension_audit(
        history=history,
        pit_used=False,
        snapshot=pl.DataFrame({
            "symbol": [AAA],
            "as_of": [date(2025, 1, 1)],
        }),
        start=D1,
    )
    assert degraded["status"] == "degraded"
    assert degraded["evidence"]

    violation_dim = {
        "name": "fake",
        "status": "violation",
        "evidence": [{
            "field": "x", "source": "t", "symbol": AAA,
            "as_of": D1.isoformat(), "known_at": D5.isoformat(),
        }],
    }
    report = pit.build_leakage_report([degraded, violation_dim])
    assert report["status"] == "violation"
    assert pit.build_leakage_report([degraded])["status"] == "degraded"
    assert pit.build_leakage_report([
        pit.constituent_dimension_audit(history=None, index_symbol=None)
    ])["status"] == "ok"


# ================================================================
# 矩阵 basic filter: 逐日 ST/市值
# ================================================================


def test_basic_filter_uses_pit_st_history_not_today_name(tmp_path):
    snap1 = _snapshot([
        (AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    snap4 = _snapshot([
        (AAA, "*ST甲", 100.0, 80.0, date(2020, 1, 1)),
        (BBB, "乙公司", 200.0, 100.0, date(2020, 1, 1)),
    ])
    pit.record_instrument_observations(tmp_path, snap1, D1)
    pit.record_instrument_observations(tmp_path, snap4, D4)
    market = pit.attach_pit_fields_to_market(
        build_market_data_matrix(_panel((AAA, BBB))),
        pit.load_instrument_history(tmp_path),
    )
    config = {"enabled": True, "exclude_st": True}
    mask = _build_basic_filter_mask_uncached(market, config)
    aaa_col = list(market.symbols).index(AAA)

    def _mask(day: date) -> bool:
        return bool(mask[_day_rows(market, day)[0], aaa_col])

    # 尽管"今天"(D4) AAA 已 ST, D2 的历史候选不被误杀; D4 起被排除。
    assert _mask(D2) is True
    assert _mask(D3) is True
    assert _mask(D4) is False


# ================================================================
# 面板路径: join_asof 注入逐日列
# ================================================================


def test_panel_attach_pit_columns_asof(tmp_path):
    snap1 = _snapshot([(AAA, "甲公司", 100.0, 80.0, date(2020, 1, 1))])
    snap4 = _snapshot([(AAA, "ST甲", 300.0, 90.0, date(2020, 1, 1))])
    pit.record_instrument_observations(tmp_path, snap1, D1)
    pit.record_instrument_observations(tmp_path, snap4, D4)

    panel = _panel((AAA,), days=6)
    enriched = pit.attach_pit_columns_panel(panel, pit.load_instrument_history(tmp_path))
    row_d2 = enriched.filter(pl.col("date") == D2).row(0, named=True)
    row_d4 = enriched.filter(pl.col("date") == D4).row(0, named=True)
    assert row_d2[PIT_IS_ST] is False
    assert row_d2[PIT_TOTAL_SHARES] == pytest.approx(100.0)
    assert row_d4[PIT_IS_ST] is True
    assert row_d4[PIT_TOTAL_SHARES] == pytest.approx(300.0)
    # 不残留临时列。
    assert not {c for c in enriched.columns if c.startswith("_pit_")}


# ================================================================
# 静态 look-ahead lint: 读路径接线回归守卫
# ================================================================


def test_backtest_read_path_is_wired_through_pit():
    engine_source = (BACKEND_DIR / "app" / "backtest" / "engine.py").read_text()
    matrix_source = (BACKEND_DIR / "app" / "backtest" / "matrix.py").read_text()
    strategy_source = (BACKEND_DIR / "app" / "backtest" / "strategy.py").read_text()

    # 矩阵入口必须解析 universe 并附加 PIT 维度/审计。
    assert "resolve_matrix_universe" in engine_source
    assert "attach_pit_fields_to_market" in engine_source
    assert "pit_audit" in engine_source
    # basic filter 必须先看 PIT 时变矩阵, 静态名称仅作降级。
    assert "PIT_IS_ST" in matrix_source
    assert "PIT_LISTED" in matrix_source and "PIT_MEMBER" in matrix_source
    # 结果必须带 universe 定义与 leakage 报告, 违例失败。
    assert "universe_definition" in strategy_source
    assert "leakage_report" in strategy_source
    assert "FutureLeakageError" in strategy_source
