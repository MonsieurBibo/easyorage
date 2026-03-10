# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "altair",
#     "openlayers",
#     "pyarrow==23.0.1",
#     "pandas==3.0.1",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import polars as pl
    import altair as alt
    import openlayers as ol
    import math

    return alt, math, mo, ol, pl


@app.cell
def load_data(pl):
    import pathlib
    DATA_PATH = str(pathlib.Path(__file__).parent.parent / "data_train_databattle2026" / "segment_alerts_all_airports_train.csv")

    N_ROWS = None

    df = pl.read_csv(
        DATA_PATH,
        n_rows=N_ROWS,
        schema_overrides={
            "lightning_id": pl.Int64,
            "lightning_airport_id": pl.Int64,
            "lon": pl.Float64,
            "lat": pl.Float64,
            "amplitude": pl.Float64,
            "maxis": pl.Float64,
            "icloud": pl.Boolean,
            "dist": pl.Float64,
            "azimuth": pl.Float64,
            "airport_alert_id": pl.Float64,
        },
    ).with_columns(
        pl.col("date").str.to_datetime("%Y-%m-%d %H:%M:%S%z").alias("date"),
    )
    return (df,)


@app.cell
def data_explorer(df, mo):
    mo.md("## Exploration libre des données")
    explorer = mo.ui.data_explorer(df)
    return (explorer,)


@app.cell
def show_explorer(explorer):
    explorer
    return


@app.cell
def basic_stats(df, mo, pl):
    desc = df.describe()

    missing = df.null_count()

    airport_counts = (
        df.group_by("airport")
        .agg(
            pl.len().alias("nb_eclairs"),
            pl.col("airport_alert_id").drop_nulls().n_unique().alias("nb_alertes"),
        )
        .sort("nb_eclairs", descending=True)
    )

    mo.output.replace(
        mo.vstack([
            mo.md("## Statistiques de base"),
            mo.md("### describe()"),
            desc,
            mo.md("### Valeurs manquantes"),
            missing,
            mo.md("### Distribution par aéroport"),
            airport_counts,
        ])
    )
    return


@app.cell
def airport_selector(df, mo):
    airports = sorted(df["airport"].unique().to_list())
    dropdown = mo.ui.dropdown(
        options={a: a for a in ["Tous"] + airports},
        value="Tous",
        label="Aéroport",
    )
    return (dropdown,)


@app.cell
def show_dropdown(dropdown, mo):
    mo.output.replace(
        mo.vstack([
            mo.md("## Filtrer par aéroport"),
            dropdown,
        ])
    )
    return


@app.cell
def filter_df(df, dropdown, pl):
    if dropdown.value == "Tous":
        df_filtered = df
    else:
        df_filtered = df.filter(pl.col("airport") == dropdown.value)
    return (df_filtered,)


@app.cell
def temporal_distribution(alt, df_filtered, mo, pl):
    temporal = (
        df_filtered.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .group_by("year", "month")
        .agg(pl.len().alias("nb_eclairs"))
        .sort("year", "month")
        .with_columns(
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("month").cast(pl.Utf8).str.pad_start(2, "0")).alias("period")
        )
    )

    chart_temporal = (
        alt.Chart(temporal.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("period:N", title="Mois", sort=None),
            y=alt.Y("nb_eclairs:Q", title="Nombre d'éclairs"),
            tooltip=["period", "nb_eclairs"],
        )
        .properties(width=800, height=300, title="Nombre d'éclairs par mois")
    )

    mo.output.replace(
        mo.vstack([
            mo.md("## Distribution temporelle"),
            chart_temporal,
        ])
    )
    return


@app.cell
def amplitude_distribution(alt, df_filtered, mo, pl):
    bin_width = 10
    amp_data = (
        df_filtered.with_columns(
            (pl.col("amplitude") / bin_width).floor() * bin_width,
            pl.when(pl.col("icloud")).then(pl.lit("IC (intra-nuage)")).otherwise(pl.lit("CG (nuage-sol)")).alias("type"),
        )
        .group_by("amplitude", "type")
        .agg(pl.len().alias("count"))
        .sort("amplitude")
    )

    chart_amp = (
        alt.Chart(amp_data.to_pandas())
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("amplitude:Q", title="Amplitude (kA)"),
            y=alt.Y("count:Q", title="Nombre d'éclairs", stack=None),
            color=alt.Color("type:N", title="Type"),
            tooltip=["type:N", "count:Q", "amplitude:Q"],
        )
        .properties(width=800, height=300, title="Distribution des amplitudes par type")
    )

    mo.output.replace(
        mo.vstack([
            mo.md("## Distribution des amplitudes"),
            chart_amp,
        ])
    )
    return


@app.cell
def spatial_map(df_filtered, mo, ol):
    sample_size = min(2_000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, seed=42)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["lon"], row["lat"]],
                },
                "properties": {
                    "amplitude": row["amplitude"],
                    "airport": row["airport"],
                    "icloud": row["icloud"],
                },
            }
            for row in df_sample.iter_rows(named=True)
        ],
    }

    source = ol.VectorSource(geojson=geojson)

    points_layer = ol.VectorLayer(
        source=source,
        style=ol.FlatStyle(
            circle_radius=3,
            circle_fill_color="rgba(255, 50, 50, 0.4)",
            circle_stroke_color="rgba(200, 0, 0, 0.6)",
            circle_stroke_width=0.5,
        ),
    )

    lon_min = df_sample["lon"].min()
    lon_max = df_sample["lon"].max()
    lat_min = df_sample["lat"].min()
    lat_max = df_sample["lat"].max()

    m = ol.MapWidget(
        layers=[ol.TileLayer(source=ol.OSM()), points_layer],
        height="500px",
    )
    m.add_control(ol.FullScreenControl())
    m.fit_bounds((lon_min, lat_min, lon_max, lat_max))

    mo.output.replace(
        mo.vstack([
            mo.md(f"## Carte des éclairs (échantillon de {sample_size} points)"),
            m,
        ])
    )
    return


@app.cell
def alert_analysis(alt, df_filtered, mo, pl):
    df_alerts = df_filtered.filter(pl.col("airport_alert_id").is_not_null())

    mo.stop(
        len(df_alerts) == 0,
        mo.md("## Analyse des alertes\n\nAucune alerte pour cette sélection."),
    )

    alert_stats = (
        df_alerts.group_by("airport_alert_id", "airport")
        .agg(
            pl.len().alias("nb_eclairs"),
            pl.col("date").min().alias("debut"),
            pl.col("date").max().alias("fin"),
            pl.col("dist").mean().alias("dist_moyenne"),
            pl.col("amplitude").mean().alias("amplitude_moyenne"),
        )
        .with_columns(
            ((pl.col("fin") - pl.col("debut")).dt.total_seconds() / 60).alias("duree_min")
        )
        .sort("nb_eclairs", descending=True)
    )

    summary = pl.DataFrame({
        "Métrique": [
            "Nombre total d'alertes",
            "Éclairs par alerte (médiane)",
            "Éclairs par alerte (moyenne)",
            "Durée médiane (min)",
            "Durée moyenne (min)",
        ],
        "Valeur": [
            str(len(alert_stats)),
            str(round(alert_stats["nb_eclairs"].median(), 1)),
            str(round(alert_stats["nb_eclairs"].mean(), 1)),
            str(round(alert_stats["duree_min"].median(), 1)),
            str(round(alert_stats["duree_min"].mean(), 1)),
        ],
    })

    chart_alert_size = (
        alt.Chart(alert_stats.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("nb_eclairs:Q", bin=alt.Bin(maxbins=50), title="Nombre d'éclairs par alerte"),
            y=alt.Y("count():Q", title="Nombre d'alertes"),
            tooltip=["count():Q"],
        )
        .properties(width=800, height=250, title="Distribution de la taille des alertes")
    )

    chart_alert_duration = (
        alt.Chart(alert_stats.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("duree_min:Q", bin=alt.Bin(maxbins=50), title="Durée (minutes)"),
            y=alt.Y("count():Q", title="Nombre d'alertes"),
            tooltip=["count():Q"],
        )
        .properties(width=800, height=250, title="Distribution de la durée des alertes")
    )

    mo.output.replace(
        mo.vstack([
            mo.md("## Analyse des alertes"),
            summary,
            chart_alert_size,
            chart_alert_duration,
        ])
    )
    return


@app.cell
def polar_scatter(alt, df_filtered, math, mo, pl):
    polar_data = df_filtered.select(
        pl.col("dist"),
        pl.col("icloud"),
        (pl.col("dist") * (pl.col("azimuth") * math.pi / 180).cos()).alias("x"),
        (pl.col("dist") * (pl.col("azimuth") * math.pi / 180).sin()).alias("y"),
    ).sample(n=min(5_000, len(df_filtered)), seed=42).with_columns(
        pl.when(pl.col("icloud")).then(pl.lit("IC")).otherwise(pl.lit("CG")).alias("type")
    )

    chart_polar = (
        alt.Chart(polar_data.to_pandas())
        .mark_circle(size=2, opacity=0.3)
        .encode(
            x=alt.X("x:Q", title="Distance Est-Ouest (km)"),
            y=alt.Y("y:Q", title="Distance Nord-Sud (km)"),
            color=alt.Color("type:N", title="Type"),
        )
        .properties(width=450, height=450, title="Répartition spatiale (projection polaire)")
    )

    mo.output.replace(
        mo.vstack([
            mo.md("## Distribution distance / azimuth"),
            chart_polar,
        ])
    )
    return


@app.cell
def dist_histogram(alt, df_filtered, mo, pl):
    dist_data = df_filtered.select(
        pl.col("dist"),
        pl.when(pl.col("icloud")).then(pl.lit("IC")).otherwise(pl.lit("CG")).alias("type"),
    ).sample(n=min(5_000, len(df_filtered)), seed=42)

    chart_dist = (
        alt.Chart(dist_data.to_pandas())
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("dist:Q", bin=alt.Bin(maxbins=30), title="Distance (km)"),
            y=alt.Y("count():Q", title="Nombre d'éclairs", stack=None),
            color=alt.Color("type:N", title="Type"),
        )
        .properties(width=600, height=250, title="Distribution des distances")
    )

    mo.output.replace(chart_dist)
    return


@app.cell
def _(df, mo, pl):
    """Réponses aux questions clés du formulaire DataBattle."""
    df_alerts_global = df.filter(pl.col("airport_alert_id").is_not_null())

    alert_stats_global = (
        df_alerts_global.group_by("airport_alert_id", "airport")
        .agg(
            pl.col("date").min().alias("debut"),
            pl.col("date").max().alias("fin"),
        )
        .with_columns(
            ((pl.col("fin") - pl.col("debut")).dt.total_seconds() / 60).alias("duree_min")
        )
    )

    nb_alertes = len(alert_stats_global)
    duree_mediane = round(alert_stats_global["duree_min"].median(), 1)

    mo.output.replace(
        mo.vstack([
            mo.md("## Réponses aux questions clés"),
            mo.callout(
                mo.md(f"""
    **Nombre d'alertes orageuses dans les données :** `{nb_alertes}`

    **Durée médiane d'une alerte :** `{duree_mediane} minutes`

    *(calculé sur l'ensemble du dataset, toutes alertes avec `airport_alert_id` non-null)*
    """),
                kind="info",
            ),
        ])
    )
    return


@app.cell
def _(mo):
    mo.output.replace(
        mo.vstack([
            mo.md("## Approche & Références"),
            mo.callout(
                mo.md("""
    ### Référence similaire

    **Shafer & Fuelberg (2019)** — *A logistic regression model for lightning cessation using the Geostationary Lightning Mapper* (Journal of Geophysical Research: Atmospheres).
    Modèle probabiliste qui classe chaque éclair comme potentiellement le dernier : approche très proche du nôtre, mais avec données GLM satellite + radar dual-pol.

    GitHub proche : [Lightning Cessation · NOAA WFO Tampa](https://github.com/thunderbolt-wx/lightning-cessation) — implémentation de règles de cessation avec données LMA/radar.
    """),
                kind="success",
            ),
            mo.callout(
                mo.md("""
    ### Approche envisagée

    **Survival analysis + gradient boosting séquentiel**

    À chaque nouvel éclair dans une alerte, on estime P(cessation dans les N prochaines minutes) à partir de features construites dynamiquement :

    - **Temporelles** : intervalle inter-éclairs, flash rate (rolling 5/10/30 min), tendance (décroissance)
    - **Spatiales** : dispersion spatiale des éclairs récents, dérive du centroïde par rapport à l'aéroport
    - **Physiques** : ratio IC/CG glissant, amplitude moyenne et sa tendance
    - **Survie** : temps depuis le début de l'alerte, dernier éclair CG connu

    Modèle principal : **XGBoost / LightGBM** avec target `is_last_lightning_cloud_ground`.
    Modèle complémentaire : **Cox Proportional Hazards** (gap identifié dans la littérature — aucune étude n'applique formellement du survival analysis à ce problème).
    """),
                kind="neutral",
            ),
            mo.callout(
                mo.md("""
    ### Évaluation

    Métriques adaptées au coût asymétrique (fausse cessation = danger réel) :

    - **FAR** (False Alarm Ratio) — priorité absolue : ne pas déclarer la cessation trop tôt
    - **POD** (Probability of Detection) — capturer les vraies cessations
    - **CSI** (Critical Success Index) = hits / (hits + misses + false alarms)
    - **Lead time moyen** : minutes gagnées vs baseline "30 minutes après le dernier éclair"

    Validation : **leave-one-airport-out cross-validation** (5 splits) pour tester la généralisation géographique.
    Baseline : règle des 30 min (POD ≈ 1.0, lead time = 0 min par définition).
    """),
                kind="warn",
            ),
        ])
    )
    return


if __name__ == "__main__":
    app.run()
