"""MetroFlow Intelligence — interactive Hangzhou historical-replay app."""
import json
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.i18n import STRINGS
from src.features import INTERVAL
from src.forecast import forecast_steps, interval_for_prediction, demand_label
from src.network_view import marker_sizes, separated_layout

ROOT = Path(__file__).parent
st.set_page_config(page_title="MetroFlow Intelligence", page_icon="🚇", layout="wide")


@st.cache_resource
def models():
    return {"inflow": joblib.load(ROOT / "artifacts/metro_xgb_model.joblib"),
            "outflow": joblib.load(ROOT / "artifacts/metro_xgb_outflow_model.joblib")}


@st.cache_data
def resources():
    frame = pd.read_csv(ROOT / "data/processed/flow_15min_split.csv", parse_dates=["interval"])
    names = pd.read_csv(ROOT / "data/station_names.csv")
    adj = pd.read_csv(ROOT / "data/Metro_roadMap.csv", index_col=0)
    adj.index = adj.index.astype(int)
    adj.columns = adj.columns.astype(int)
    report = json.loads((ROOT / "reports/evaluation.json").read_text(encoding="utf-8"))
    return frame, dict(zip(names.stationID.astype(int), names.name)), adj, report


def history_at(frame, station, as_of, target):
    observed = frame[(frame.stationID == station) & (frame.interval <= as_of)].tail(4)
    if (len(observed) != 4 or
            list(observed.interval) != [as_of - INTERVAL * i for i in (3, 2, 1, 0)]):
        return None
    return observed[target].to_numpy(dtype=float).tolist()


@st.cache_data
def combined_thresholds(frame):
    training = frame.loc[frame.split == "train"].copy()
    training["both"] = training.inflow + training.outflow
    result = training.groupby("stationID").both.quantile([.5, .9]).unstack()
    return {int(i): {"medium": float(r[.5]), "high": float(r[.9])}
            for i, r in result.iterrows()}


def network_predictions(frame, artifacts, as_of, target, steps):
    result = {}
    total_limits = combined_thresholds(frame) if target == "both" else None
    for station in frame.stationID.unique():
        station = int(station)
        predictions = {}
        for direction in (["inflow", "outflow"] if target == "both" else [target]):
            values = history_at(frame, station, as_of, direction)
            if values is None:
                break
            predictions[direction] = forecast_steps(artifacts[direction], station,
                                                    as_of, values, steps)[-1]
        if len(predictions) != (2 if target == "both" else 1):
            continue
        value = sum(predictions.values())
        limits = (total_limits.get(station) if target == "both" else
                  artifacts[target]["station_thresholds"].get(str(station)))
        if limits:
            result[station] = {"value": value,
                               "level": demand_label(value, limits["medium"], limits["high"]),
                               **predictions}
    return result


def network_figure(adj, names, values, selected, title, t):
    graph = nx.from_pandas_adjacency(adj)
    sizes = marker_sizes(values, selected)
    pos = separated_layout(graph, sizes, selected)
    figure = go.Figure()
    xx, yy, focus_x, focus_y = [], [], [], []
    for start, end in graph.edges():
        pair_x = [pos[start][0], pos[end][0], None]
        pair_y = [pos[start][1], pos[end][1], None]
        if selected in (int(start), int(end)):
            focus_x.extend(pair_x)
            focus_y.extend(pair_y)
        else:
            xx.extend(pair_x)
            yy.extend(pair_y)
    figure.add_trace(go.Scatter(x=xx, y=yy, mode="lines", line=dict(color="#7c8da4", width=1.35),
                                hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(x=focus_x, y=focus_y, mode="lines",
                                line=dict(color="#fbbf24", width=2.7),
                                hoverinfo="skip", showlegend=False))
    colors = {"low": "#20a47f", "medium": "#e8a13b", "high": "#db5863", "missing": "#94a3b8"}
    for level in colors:
        ids = [int(i) for i in graph.nodes if int(i) != selected and
               values.get(int(i), {}).get("level", "missing") == level]
        if not ids:
            continue
        figure.add_trace(go.Scatter(
            x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids], mode="markers",
            text=[names.get(i, f"#{i}") for i in ids],
            customdata=[[i, values.get(i, {}).get("value", float("nan"))] for i in ids],
            hovertemplate=f"%{{text}}<br>%{{customdata[1]:.0f}} {t['unit_short']}<extra></extra>",
            marker=dict(color=colors[level], size=[sizes.get(i, 10) for i in ids],
                        opacity=.7, line=dict(color="white", width=1.1)), name=t.get(level, t["not_available"])))
    if selected in pos:
        value = values.get(selected, {})
        label = names.get(selected, str(selected))
        figure.add_trace(go.Scatter(x=[pos[selected][0]], y=[pos[selected][1]],
                                    mode="markers+text", text=[f"★ {label}"],
                                    textposition="top center", textfont=dict(size=16, color="#fef3c7"),
                                    customdata=[[selected, value.get("value", float("nan"))]],
                                    hovertemplate=f"%{{text}}<br>%{{customdata[1]:.0f}} {t['unit_short']}<extra></extra>",
                                    marker=dict(size=max(34, sizes.get(selected, 10)+12),
                                                color=colors.get(value.get("level"), "#94a3b8"),
                                                line=dict(color="#fef08a", width=5)),
                                    name=t["selected"], showlegend=False))
    figure.update_layout(title=title, height=740, margin=dict(l=5, r=5, t=55, b=45),
                         xaxis=dict(visible=False, range=[0, 1350], fixedrange=False),
                         yaxis=dict(visible=False, range=[0, 710], fixedrange=False, scaleanchor="x", scaleratio=1),
                         legend=dict(orientation="h", y=-.05), hovermode="closest", dragmode="select")
    return figure


current_lang = st.session_state.get("language_choice", "English")
language = st.sidebar.selectbox(STRINGS[current_lang]["language"], list(STRINGS),
                                key="language_choice")
t = STRINGS[language]
st.title("🚇 MetroFlow Intelligence")
st.caption(t["subtitle"])

frame, station_names, adjacency, report = resources()
artifacts = models()
with st.sidebar:
    st.divider()
    target = st.radio(t["direction"], ["inflow", "outflow", "both"],
                      format_func=lambda x: t[x], horizontal=True)
    minutes = st.segmented_control(t["horizon"], [15, 30, 60], default=15,
                                   format_func=lambda x: f"+{x} {t['minutes_short']}")
    steps = minutes // 15
    dates = sorted(frame.interval.dt.date.unique())
    selected_date = st.selectbox(t["date"], dates, index=min(10, len(dates)-1))
    possible_times = sorted(pd.Timestamp(x) for x in frame.loc[
        frame.interval.dt.date == selected_date, "interval"].unique())
    as_of = pd.Timestamp(st.select_slider(t["time"], options=possible_times,
                                              value=possible_times[len(possible_times)//2],
                                              format_func=lambda x: x.strftime("%H:%M")))
    if "pending_station" in st.session_state:
        st.session_state.selected_station = st.session_state.pop("pending_station")
    station = st.selectbox(t["station"], sorted(station_names), key="selected_station",
                           format_func=lambda sid: f"{station_names[sid]} (#{sid})")

forecast_tab, map_tab, quality_tab, about_tab = st.tabs(
    [t["forecast"], t["map"], t["quality"], t["about"]])

with forecast_tab:
    directions = ["inflow", "outflow"] if target == "both" else [target]
    histories = {direction: history_at(frame, station, as_of, direction)
                 for direction in directions}
    if any(values is None for values in histories.values()):
        st.info(t["history_error"])
    else:
        predictions_by_direction = {
            direction: forecast_steps(artifacts[direction], station, as_of, histories[direction], steps)
            for direction in directions}
        future_dates = [as_of + i * INTERVAL for i in range(1, steps+1)]
        columns = st.columns(len(directions) + 1)
        for index, direction in enumerate(directions):
            forecast = predictions_by_direction[direction][-1]
            lower, upper = interval_for_prediction(
                forecast, artifacts[direction]["calibration_error_p90"], steps)
            columns[index].metric(t[direction], f"{forecast:,.0f}", t["units"], delta_color="off")
            columns[index].caption(f"{t['range']}: {lower:,.0f}–{upper:,.0f}")
        total = sum(pred[-1] for pred in predictions_by_direction.values())
        station_limits = (combined_thresholds(frame)[station] if target == "both" else
                          artifacts[target]["station_thresholds"][str(station)])
        level = demand_label(total, station_limits["medium"], station_limits["high"])
        columns[-1].metric(t["total"] if target == "both" else t["demand"],
                           f"{total:,.0f}" if target == "both" else t[level])
        if target == "both":
            st.caption(f"{t['demand']}: {t[level]}")
        st.caption(t["relative"])

        actual = frame[(frame.stationID == station) & (frame.interval <= as_of)].tail(16)
        fig = go.Figure()
        palette = {"inflow": "#38bdf8", "outflow": "#a78bfa"}
        for direction in directions:
            fig.add_trace(go.Scatter(x=actual.interval, y=actual[direction], mode="lines+markers",
                                     name=f"{t[direction]} · {t['history']}",
                                     line=dict(color=palette[direction])))
            fig.add_trace(go.Scatter(x=[as_of, *future_dates],
                                     y=[histories[direction][-1], *predictions_by_direction[direction]],
                                     mode="lines+markers", name=f"{t[direction]} · {t['predicted']}",
                                     line=dict(color=palette[direction], dash="dash")))
        fig.update_layout(title=t["chart"], yaxis_title=t["units"], height=380)
        st.plotly_chart(fig, width="stretch")
        st.info(f"**{t['operator']}:** {t['advice_'+level]}")
        export = pd.DataFrame({t["id_col"]: station, t["station_col"]: station_names[station],
                               t["time"]: as_of, t["forecast_date"]: future_dates})
        for direction in directions:
            export[t[direction]] = predictions_by_direction[direction]
        if target == "both":
            export[t["total"]] = [sum(predictions_by_direction[d][j] for d in directions)
                                   for j in range(steps)]
        st.download_button(t["download"], data=export.to_csv(index=False),
                           file_name="metroflow_forecast.csv", mime="text/csv")
        st.caption(t["uncertainty"])

with map_tab:
    st.subheader(t["maptitle"])
    st.caption(t["select_tip"])
    st.caption(t["percentile_info"])
    with st.spinner(t["map_loading"]):
        predictions_map = network_predictions(frame, artifacts, as_of, target, steps)
    st.metric(t["valid"], len(predictions_map))
    fig = network_figure(adjacency, station_names, predictions_map, station,
                         f"{t[target]} • {as_of + minutes * pd.Timedelta(minutes=1):%Y-%m-%d %H:%M}", t)
    picked = st.plotly_chart(fig, width="stretch", config={"scrollZoom": True},
                             on_select="rerun", selection_mode="points", key="network_pick")
    selected_points = picked.selection.points if picked is not None else []
    if selected_points:
        payload = selected_points[0].get("customdata")
        if payload:
            picked_id = int(payload[0])
            if picked_id != st.session_state.get("last_map_station"):
                st.session_state.last_map_station = picked_id
                if picked_id in station_names and picked_id != station:
                    st.session_state.pending_station = picked_id
                    st.rerun()
    if predictions_map:
        rows = []
        for sid, entry in predictions_map.items():
            row = {t["station_col"]: station_names.get(sid, str(sid)),
                   t["id_col"]: sid, t["count_col"]: round(entry["value"]),
                   t["relative_col"]: t[entry["level"]]}
            if target == "both":
                row[t["inflow"]] = round(entry["inflow"])
                row[t["outflow"]] = round(entry["outflow"])
            rows.append(row)
        table = pd.DataFrame(rows).sort_values(t["count_col"], ascending=False).reset_index(drop=True)
        st.subheader(t["table_title"])
        st.caption(t["table_hint"])
        def selected_style(row):
            focused = int(row[t["id_col"]]) == station
            style = "background-color: #503829; color: #fff3d6; font-weight: bold" if focused else ""
            return [style] * len(row)
        table_key = f"table_{target}_{minutes}_{as_of.isoformat()}"
        event = st.dataframe(table.style.apply(selected_style, axis=1),
                             hide_index=True, width="stretch", height=470,
                             selection_mode="single-row", on_select="rerun", key=table_key)
        selected_rows = event.selection.rows if event is not None else []
        if selected_rows:
            picked_id = int(table.iloc[selected_rows[0]][t["id_col"]])
            signature = (table_key, picked_id)
            if signature != st.session_state.get("last_table_selection"):
                st.session_state.last_table_selection = signature
                if picked_id != station:
                    st.session_state.pending_station = picked_id
                    st.rerun()
    st.caption(t["relative"])

with quality_tab:
    st.subheader(t["report"])
    st.caption(t["test"])
    for direction in (["inflow", "outflow"] if target == "both" else [target]):
        st.markdown(f"### {t[direction]}")
        scores = report["targets"][direction]
        c1, c2, c3 = st.columns(3)
        c1.metric(t["mae"], f"{scores['test']['mae']:.2f}")
        c2.metric(t["wape"], f"{scores['test_wape']:.1%}")
        c3.metric(t["coverage"], f"{scores['test_interval_coverage_at_q90']:.1%}")
        st.caption(t["comparison"])
        st.bar_chart(pd.DataFrame({"name": [t["baseline"], "XGBoost"],
                                   "MAE": [scores["test_baseline"]["mae"], scores["test"]["mae"]]})
                     .set_index("name"), x_label=t["model_label"])
        st.caption(t["period"])
        period_labels = {"morning": t["peak_morning"], "evening": t["peak_evening"],
                         "offpeak": t["peak_offpeak"]}
        st.bar_chart(pd.Series({period_labels[key]: value for key, value in
                                scores["peak_test_mae"].items()}, name="MAE"))

with about_tab:
    st.subheader(t["source_heading"])
    st.write(t["source"])
    st.markdown(f"[{t['source_link']}](https://doi.org/10.5281/zenodo.3145404)")
    st.write(t["privacy"])
    st.write(t["limits"])
