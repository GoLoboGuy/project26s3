#
# ===============================================
#   Stock Price Visualizer v4
# ===============================================
#
# ---------- [ v2 Features ] ----------
# 1) 지표 프리셋(단타/스윙/중장기)
# 2) RSI 과매수/과매도 영역 강조
# 3) MACD 히스토그램 색상 분기
# 4) 누적 수익률 라인 차트
#
# ---------- [ v3 Features ] ----------
# 1) 차트 옵션 커스터마이징
#    - 이동평균선 기간/개수 선택
#    - 이동평균선 색상 선택
#    - 차트 타입 변경 (candle/ohlc/line)
#    - 배경색 변경 (white/black)
# 2) 거래량 시각화 개선
#    - 캔들차트와 분리하여 "별도 거래량 차트" 출력
#    - 거래량 급증일(평균 대비 n배 이상) 강조 표시
# 3) 기술적 지표 추가 (ON/OFF)
#    - RSI (pandas 직접 계산)
#    - MACD (pandas 직접 계산)
#    - Bollinger Bands
# 4) 수익률 분석
#    - 선택 기간 수익률(%)
#    - 최고/최저가
#    - 변동성(일간 표준편차, 연환산 표준편차)
#
# ---------- [ v4 Features ] ----------
# 1) RSI 30/70 영역 밴드 색칠
# 2) UI/UX 강화
#     - 오늘의 신호(강한주의/주의/중립/긍정/강한긍정) 배지 출력
#     - 난이도(초급/중급) 차트요약 설명
#     - 용어 도움말(tooltip)
# ===============================================
#

# =========================
# 라이브러리 호출
# =========================
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import FinanceDataReader as fdr
import mplfinance as mpf


# =========================
# 기본 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
MIN_DAYS = 5

PRESETS = {
    "단타": {"mav": (5, 10), "rsi": 7, "macd": (6, 13, 5)},
    "스윙": {"mav": (5, 20, 60), "rsi": 14, "macd": (12, 26, 9)},
    "중장기": {"mav": (20, 60, 120), "rsi": 21, "macd": (12, 26, 9)},
}


# =========================
# 유틸
# =========================
def load_json(filename: str) -> dict:
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 데이터 로딩(캐시)
# =========================
@st.cache_data(show_spinner=False)
def get_symbols(market: str = "KOSPI") -> pd.DataFrame:
    df = fdr.StockListing(market)
    if df is None or df.empty:
        return pd.DataFrame(columns=["Code", "Name", "Market"])
    if "Marcap" in df.columns:
        df.sort_values("Marcap", ascending=False, inplace=True)
    return df[["Code", "Name", "Market"]]


@st.cache_data(show_spinner=False)
def load_stock_data(code: str, days: int) -> pd.DataFrame:
    if not isinstance(days, int):
        raise ValueError("기간(days)은 정수여야 합니다.")
    if days < MIN_DAYS:
        raise ValueError(f"분석 기간은 최소 {MIN_DAYS}일 이상이어야 합니다.")

    start = (datetime.today() - timedelta(days=days)).date()
    end = datetime.today().date()

    df = fdr.DataReader(code, start, end)
    if df is None or df.empty:
        raise ValueError("선택한 기간에 데이터가 없습니다. 기간을 늘려보세요.")

    if "Change" in df.columns:
        df = df.drop(columns=["Change"])

    need_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not need_cols.issubset(set(df.columns)):
        raise ValueError("필수 컬럼(Open/High/Low/Close/Volume)이 부족해 차트를 그릴 수 없습니다.")

    return df


# =========================
# 지표 계산 (pandas)
# =========================
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def add_indicators(df: pd.DataFrame, preset_name: str) -> pd.DataFrame:
    cfg = PRESETS.get(preset_name, PRESETS["스윙"])
    rsi_period = int(cfg["rsi"])
    macd_fast, macd_slow, macd_signal = map(int, cfg["macd"])

    out = df.copy()

    out["RSI"] = calc_rsi(out["Close"], period=rsi_period)
    out["MACD"], out["MACD_SIG"], out["MACD_HIST"] = calc_macd(
        out["Close"], fast=macd_fast, slow=macd_slow, signal=macd_signal
    )

    # 추세 판단용 이평(요약/배지용)
    out["MA5"] = out["Close"].rolling(5).mean()
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA60"] = out["Close"].rolling(60).mean()

    # 수익률
    out["RET_D1"] = out["Close"].pct_change()
    out["CUMRET"] = (1 + out["RET_D1"]).cumprod() - 1

    return out


# =========================
# 5단계 배지 판단
# =========================
def judge_signal_badge_5(df: pd.DataFrame) -> dict:
    """
    규칙 기반 5단계 배지
    - 강한주의:
        * 하락 추세(5<20<60) AND MACD 히스토그램 음수 AND RSI<=35
        * 또는 RSI<=25 (극단 과매도)
    - 주의:
        * RSI>=70(과열) 또는 RSI<=30(침체)
        * 또는 하락 추세 + MACD 음수
    - 강한긍정:
        * 상승 추세(5>20>60) AND MACD 히스토그램 양수 & 증가 AND RSI 45~69
        * 그리고 전일 수익률 양수(추세 + 탄력)
    - 긍정:
        * 상승 추세 AND (MACD 양수 or 개선) AND RSI 과열 아님
    - 중립: 나머지
    """
    if df is None or df.empty or len(df) < 2:
        return {"level": "중립", "reason": "데이터가 부족해 중립으로 표시합니다."}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 추세
    ma5, ma20, ma60 = last["MA5"], last["MA20"], last["MA60"]
    trend_ok = pd.notna(ma5) and pd.notna(ma20) and pd.notna(ma60)
    trend_up = trend_ok and (ma5 > ma20 > ma60)
    trend_down = trend_ok and (ma5 < ma20 < ma60)

    # RSI
    rsi_v = last["RSI"]
    rsi_ok = pd.notna(rsi_v)
    rsi_over = rsi_ok and (rsi_v >= 70)
    rsi_under = rsi_ok and (rsi_v <= 30)
    rsi_extreme_under = rsi_ok and (rsi_v <= 25)

    # MACD 모멘텀
    hist = last["MACD_HIST"]
    hist_prev = prev["MACD_HIST"]
    hist_ok = pd.notna(hist) and pd.notna(hist_prev)
    macd_pos = hist_ok and (hist >= 0)
    macd_pos_inc = hist_ok and (hist >= 0) and (hist > hist_prev)
    macd_neg = hist_ok and (hist < 0)

    # 전일 등락
    d1 = last["RET_D1"]
    d1_ok = pd.notna(d1)
    up_today = d1_ok and (d1 > 0)

    # ---- 강한주의 ----
    if rsi_extreme_under:
        return {"level": "강한주의", "reason": f"RSI {float(rsi_v):.1f}로 매우 낮아 극단적 과매도 구간입니다."}

    if trend_down and macd_neg and rsi_ok and (rsi_v <= 35):
        return {"level": "강한주의", "reason": "하락 추세 + 하락 모멘텀 + RSI 약세가 동시에 나타납니다."}

    # ---- 주의 ----
    if rsi_over:
        return {"level": "주의", "reason": f"RSI {float(rsi_v):.1f}로 과매수(단기 과열 가능)입니다."}
    if rsi_under:
        return {"level": "주의", "reason": f"RSI {float(rsi_v):.1f}로 과매도(단기 침체 가능)입니다."}
    if trend_down and macd_neg:
        return {"level": "주의", "reason": "하락 추세와 하락 모멘텀이 우세합니다."}

    # ---- 강한긍정 ----
    if trend_up and macd_pos_inc and rsi_ok and (45 <= rsi_v <= 69) and up_today:
        return {"level": "강한긍정", "reason": "상승 추세 + 상승 모멘텀 증가 + RSI 안정 + 전일 대비 상승입니다."}

    # ---- 긍정 ----
    if trend_up and (macd_pos or macd_pos_inc) and (not rsi_over):
        return {"level": "긍정", "reason": "상승 추세가 우세하며 모멘텀이 나쁘지 않습니다(과열은 아님)."}

    # ---- 중립 ----
    return {"level": "중립", "reason": "추세/모멘텀이 혼조이거나 뚜렷한 신호가 없습니다."}


# =========================
# 요약(난이도 반영, 문장 자연화)
# =========================
def build_insights(df: pd.DataFrame, difficulty: str, badge: dict) -> dict:
    if df is None or df.empty or len(df) < 2:
        return {"headline": "데이터가 부족해 요약을 만들 수 없습니다.", "bullets": []}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 추세
    trend = "판단 불가"
    if pd.notna(last["MA5"]) and pd.notna(last["MA20"]) and pd.notna(last["MA60"]):
        if last["MA5"] > last["MA20"] > last["MA60"]:
            trend = "상승 추세(단기 > 중기 > 장기)"
        elif last["MA5"] < last["MA20"] < last["MA60"]:
            trend = "하락 추세(단기 < 중기 < 장기)"
        else:
            trend = "횡보/혼조(이평선 엇갈림)"

    # RSI 문장
    rsi_v = float(last["RSI"]) if pd.notna(last["RSI"]) else None
    if rsi_v is None:
        rsi_text = "RSI는 데이터 부족으로 계산이 어렵습니다."
    elif rsi_v >= 70:
        rsi_text = f"RSI {rsi_v:.1f}로 과열 구간에 가깝습니다(단기 과매수)."
    elif rsi_v <= 30:
        rsi_text = f"RSI {rsi_v:.1f}로 침체 구간에 가깝습니다(단기 과매도)."
    else:
        rsi_text = f"RSI {rsi_v:.1f}로 과열/침체가 아닌 중립 범위입니다."

    # MACD 문장
    hist = last["MACD_HIST"]
    hist_prev = prev["MACD_HIST"]
    if pd.isna(hist) or pd.isna(hist_prev):
        macd_text = "MACD는 데이터 부족으로 판단이 어렵습니다."
    else:
        direction = "개선" if hist > hist_prev else "약화" if hist < hist_prev else "유지"
        if hist >= 0:
            macd_text = f"MACD 모멘텀은 상승 쪽이 우세하며, 흐름은 {direction} 중입니다."
        else:
            macd_text = f"MACD 모멘텀은 하락 쪽이 우세하며, 흐름은 {direction} 중입니다."

    # 변동성
    daily_ret = df["RET_D1"].dropna()
    vol = float(daily_ret.std() * 100) if len(daily_ret) > 5 else None
    if vol is None:
        vol_text = "변동성은 데이터 부족으로 계산이 어렵습니다."
    else:
        level = "큰 편" if vol >= 3.0 else "보통" if vol >= 1.5 else "작은 편"
        vol_text = f"최근 변동성(일간 표준편차)은 {vol:.2f}%로 {level}입니다."

    # 전일 등락
    d1 = float(last["RET_D1"] * 100) if pd.notna(last["RET_D1"]) else None
    d1_text = "전일 대비 등락은 계산할 수 없습니다." if d1 is None else f"오늘 종가는 전일 대비 {d1:+.2f}%입니다."

    # 헤드라인(배지 기반)
    headline = f"오늘의 신호는 **{badge['level']}** 입니다. ({badge['reason']})"

    if difficulty == "초급":
        bullets = [
            f"추세: {trend}",
            rsi_text,
            macd_text,
            d1_text,
        ]
    else:
        bullets = [
            f"추세 판단: {trend} (MA5/MA20/MA60 정렬 기준)",
            f"{rsi_text}  ※ RSI는 70↑ 과열, 30↓ 침체로 해석합니다.",
            f"{macd_text}  ※ 히스토그램이 0 위면 상승 힘, 0 아래면 하락 힘으로 봅니다.",
            f"{vol_text}",
            f"{d1_text}",
        ]

    return {"headline": headline, "bullets": bullets}


# =========================
# UI: 배지 표시
# =========================
def render_badge(level: str):
    styles = {
        "강한주의": {"bg": "#7f1d1d", "fg": "#ffffff", "label": "강한주의"},
        "주의": {"bg": "#fee2e2", "fg": "#991b1b", "label": "주의"},
        "중립": {"bg": "#e5e7eb", "fg": "#374151", "label": "중립"},
        "긍정": {"bg": "#d1fae5", "fg": "#065f46", "label": "긍정"},
        "강한긍정": {"bg": "#065f46", "fg": "#ffffff", "label": "강한긍정"},
    }
    s = styles.get(level, styles["중립"])
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 12px;
            border-radius:999px;
            background:{s['bg']};
            color:{s['fg']};
            font-weight:800;
            font-size:14px;
            margin:4px 0 10px 0;
        ">
            오늘의 신호: {s['label']}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 렌더링
# =========================
def render_header():
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            lottie = load_json("lottie-stock-candle-loading.json")
            from streamlit_lottie import st_lottie
            st_lottie(lottie, width=150, height=150)
        except Exception:
            pass
    with col2:
        st.title("📈 주식 정보 시각화")
        # st.caption("5단계 신호 배지 + RSI 밴드 + 초보자 요약(난이도/도움말)")


def render_sidebar(symbol_choices: list[str]) -> dict:
    with st.sidebar:
        st.header("분석 설정")
        choice = st.selectbox("종목 선택", symbol_choices)
        code = choice.split()[0]

        preset = st.selectbox("지표 프리셋", list(PRESETS.keys()), index=1)
        ndays = st.slider("분석 기간(days)", MIN_DAYS, 365, 90, 1)
        chart_type = st.selectbox("차트 타입", ["candle", "ohlc", "line"], index=0)

        difficulty = st.selectbox("설명 난이도", ["초급", "중급"], index=0)

        return {
            "code": code,
            "choice_label": choice,
            "preset": preset,
            "ndays": int(ndays),
            "chart_type": chart_type,
            "difficulty": difficulty,
        }


def render_metrics(df: pd.DataFrame):
    close = df["Close"].dropna()
    if len(close) < 2:
        st.warning("수익률 계산을 위한 데이터가 부족합니다.")
        return

    period_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    daily_vol = df["RET_D1"].dropna().std() * 100 if df["RET_D1"].dropna().shape[0] > 5 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("기간 수익률(%)", f"{period_return:.2f}%")
    c2.metric("최고가", f"{df['High'].max():.0f}")
    c3.metric("최저가", f"{df['Low'].min():.0f}")
    c4.metric("변동성(일간, %)", "-" if np.isnan(daily_vol) else f"{daily_vol:.2f}%")


def render_price_chart(df: pd.DataFrame, preset_name: str, chart_type: str):
    cfg = PRESETS.get(preset_name, PRESETS["스윙"])
    mav = tuple(cfg["mav"])

    apds = []

    # RSI 패널 + 30/70 밴드
    rsi = df["RSI"]
    apds.append(mpf.make_addplot(rsi, panel=1, ylabel="RSI", width=1))
    s70 = pd.Series(70, index=df.index)
    s30 = pd.Series(30, index=df.index)
    apds.append(mpf.make_addplot(s70, panel=1, linestyle="--", width=1))
    apds.append(mpf.make_addplot(s30, panel=1, linestyle="--", width=1))

    # 밴드(조건부 fill)
    try:
        apds.append(
            mpf.make_addplot(
                rsi,
                panel=1,
                color="none",
                fill_between=dict(y1=70, y2=rsi, where=(rsi >= 70), alpha=0.20),
            )
        )
        apds.append(
            mpf.make_addplot(
                rsi,
                panel=1,
                color="none",
                fill_between=dict(y1=30, y2=rsi, where=(rsi <= 30), alpha=0.20),
            )
        )
    except Exception:
        pass

    # MACD 패널(색상 분기)
    hist = df["MACD_HIST"]
    colors = np.where(hist.fillna(0).to_numpy() >= 0, "green", "red")
    apds.append(mpf.make_addplot(hist, panel=2, type="bar", color=colors, alpha=0.6, ylabel="MACD"))
    apds.append(mpf.make_addplot(df["MACD"], panel=2, width=1))
    apds.append(mpf.make_addplot(df["MACD_SIG"], panel=2, width=1))

    fig, _ = mpf.plot(
        df,
        type=chart_type,
        mav=mav,
        addplot=apds,
        volume=True,
        panel_ratios=(6, 2, 2),
        figsize=(12, 8),
        returnfig=True,
    )
    st.pyplot(fig, clear_figure=True)


def render_term_help():
    st.subheader("❓ 용어 도움말")
    with st.expander("RSI / MACD가 뭐예요? (눌러서 보기)", expanded=False):
        st.markdown(
            """
- **RSI(Relative Strength Index)**  
  최근 일정 기간의 상승/하락 강도를 수치화한 지표입니다.  
  보통 **70 이상은 과열(과매수)**, **30 이하는 침체(과매도)**로 해석합니다.

- **MACD(Moving Average Convergence Divergence)**  
  두 개의 EMA(지수이평) 차이를 이용해 **추세의 힘(모멘텀)**을 봅니다.  
  **히스토그램이 0 위/아래**에 있는지로 상승/하락 힘의 우위를 빠르게 판단합니다.
            """
        )


def render_summary_panel(insight: dict, badge: dict):
    st.subheader("🧾 차트 요약 설명")
    render_badge(badge["level"])

    st.markdown(insight["headline"])

    with st.expander("자세한 요약 보기", expanded=True):
        for b in insight.get("bullets", []):
            st.markdown(f"- {b}")
        st.caption("※ 본 설명은 교육용이며 투자 조언이 아닙니다. 지표는 후행할 수 있습니다.")


def render_footer():
    st.divider()
    st.caption("팁) 초보자는 ‘추세 → RSI → MACD → 변동성’ 순서로 보면 이해가 빠릅니다.")


# =========================
# main
# =========================
def main():
    st.set_page_config(page_title="Stock Visualizer v4_2", layout="wide")
    render_header()

    symbols = get_symbols()
    if symbols is None or symbols.empty:
        st.error("종목 리스트를 불러오지 못했습니다. 네트워크 상태를 확인해 주세요.")
        st.stop()

    symbol_choices = [" : ".join(x) for x in zip(symbols.Code, symbols.Name, symbols.Market)]
    ui = render_sidebar(symbol_choices)

    try:
        with st.spinner("주식 데이터를 불러오는 중입니다..."):
            df = load_stock_data(ui["code"], ui["ndays"])
            df = add_indicators(df, ui["preset"])

        if len(df) < 10:
            st.warning("데이터가 너무 적어 지표 해석이 부정확할 수 있습니다. 기간을 늘려보세요.")

        st.markdown(f"<h3 style='text-align:center;'>{ui['choice_label']}</h3>", unsafe_allow_html=True)

        badge = judge_signal_badge_5(df)

        render_metrics(df)
        render_price_chart(df, ui["preset"], ui["chart_type"])

        insight = build_insights(df, ui["difficulty"], badge)
        render_summary_panel(insight, badge)

        render_term_help()

    except ValueError as e:
        st.warning(str(e))
    except Exception as e:
        st.error("데이터를 불러오는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        st.code(str(e))

    render_footer()


if __name__ == "__main__":
    main()
