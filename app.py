import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Metro Flow Forecast", page_icon="🚇")

@st.cache_resource
def load_models():
    return joblib.load("artifacts/metro_xgb_model.joblib"), joblib.load("artifacts/metro_xgb_outflow_model.joblib")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/flow_features.csv", parse_dates=["interval"])

@st.cache_data
def load_names():
    n = pd.read_csv("data/station_names.csv")
    return dict(zip(n["stationID"], n["name"]))

station_names = load_names()

inflow_art, outflow_art = load_models()
df = load_data()

# columns any model may need — used to validate a selected interval has full history
FEATURE_UNION = sorted(set(inflow_art["features"]) | set(outflow_art["features"]))

st.title("🚇 Metro Passenger Flow Forecast")
st.write("Predict passenger flow for the next 15-minute interval at a station.")

# mode selector
mode = st.radio("Show:", ["Inflow only", "Outflow only", "Both"], horizontal=True)

station = st.selectbox(
    "Station",
    sorted(df["stationID"].unique()),
    format_func=lambda sid: f"{station_names.get(sid, sid)} (#{sid})"
)
station_rows = df[df["stationID"] == station].sort_values("interval")

if station_rows.empty:
    st.warning("No flow data available for this station. Please choose another station.")
    st.stop()

times = station_rows["interval"].dt.strftime("%Y-%m-%d %H:%M").tolist()
chosen = st.selectbox("Current time", times, index=len(times)//2)

def band(p):
    if p < 100:   return "🟢 Low"
    elif p < 300: return "🟡 Medium"
    else:         return "🔴 High"

def crowd_color(p):
    if p is None: return "#cccccc"
    if p < 100:   return "#27ae60"
    if p < 300:   return "#f1c40f"
    return "#e74c3c"

# decide which directions to show
show = []
if mode in ("Inflow only", "Both"):
    show.append(("inflow", "Inflow", inflow_art, "#2980b9"))
if mode in ("Outflow only", "Both"):
    show.append(("outflow", "Outflow", outflow_art, "#8e44ad"))

if st.button("Predict next 15 min"):
    chosen_ts = pd.to_datetime(chosen)
    row = station_rows[station_rows["interval"] == chosen_ts]
    next_ts = chosen_ts + pd.Timedelta(minutes=15)

    # input validation: the selected interval must exist and be complete
    if row.empty:
        st.error("No data for the selected time. Please pick another interval.")
        st.stop()
    if not show:
        st.info("Select at least one direction (inflow or outflow) to forecast.")
        st.stop()
    if row[FEATURE_UNION].isnull().any().any():
        st.warning(
            "This interval is too early in the day to have the required history "
            "(lag features). Please choose a later time."
        )
        st.stop()

    # metric cards
    cols = st.columns(len(show))
    preds = {}
    try:
        for col, (tcol, name, art, color) in zip(cols, show):
            p = float(art["model"].predict(row[art["features"]])[0])
            preds[tcol] = p
            with col:
                st.metric(f"Predicted {name}", f"{p:.0f}")
                st.write(f"Crowding: **{band(p)}**")
                st.caption(f"(Actual: {int(row[tcol].iloc[0])})")
    except Exception as e:
        st.error(f"Prediction failed for this input: {e}")
        st.stop()

    # recent-flow chart
    history = station_rows[station_rows["interval"] <= chosen_ts].tail(12)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for tcol, name, art, color in show:
        ax.plot(history["interval"], history[tcol], marker="o", label=name, color=color)
        ax.scatter([next_ts], [preds[tcol]], color=color, s=140, edgecolor="black", zorder=5)
    ax.set_title(f"{station_names.get(station, station)} (#{station}) — recent flow with forecast")
    ax.set_ylabel("passengers"); ax.set_xlabel("time"); ax.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    st.pyplot(fig)

    # network map(s)
   # network map(s)
    all_now = df[df["interval"] == chosen_ts].copy()
    if not all_now.empty:
        import matplotlib.patches as mpatches

        adj = pd.read_csv("data/Metro_roadMap.csv", index_col=0)
        adj.columns = adj.columns.astype(int)
        G = nx.from_pandas_adjacency(adj)
        # more spacing: higher k pushes nodes apart, more iterations settles them
        pos = nx.spring_layout(G, seed=42, k=0.9, iterations=200)

        legend_handles = [
            mpatches.Patch(color="#27ae60", label="Low (<100)"),
            mpatches.Patch(color="#f1c40f", label="Medium (100–300)"),
            mpatches.Patch(color="#e74c3c", label="High (>300)"),
            mpatches.Patch(color="#cccccc", label="No data"),
        ]

        for tcol, name, art, color in show:
            all_now["pred"] = art["model"].predict(all_now[art["features"]])
            pmap = dict(zip(all_now["stationID"], all_now["pred"]))
            colors = [crowd_color(pmap.get(int(n))) for n in G.nodes()]
            # name labels instead of raw IDs
            labels = {n: station_names.get(int(n), str(n)) for n in G.nodes()}

            fig2, ax2 = plt.subplots(figsize=(16, 12))
            nx.draw_networkx_edges(G, pos, ax=ax2, edge_color="#bbbbbb", width=1.2)
            nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=colors,
                                   node_size=1400, edgecolors="#222222", linewidths=1.2)
            nx.draw_networkx_labels(G, pos, ax=ax2, labels=labels, font_size=8,
                                    font_color="black", font_weight="bold")
            ax2.set_title(f"Network {name} crowding — {chosen}", fontsize=15, fontweight="bold")
            ax2.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)
            ax2.axis("off")
            plt.tight_layout()
            st.pyplot(fig2)
