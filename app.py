"""
app.py — Interfaz Streamlit para diagnóstico de anemia por frotis sanguíneo
Grupo 7 · TPI AneRBC
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import pickle
import os
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import base64
import io

# ─────────────────────────────────────────────
# Importaciones del pipeline del notebook
# ─────────────────────────────────────────────
from pipeline import (
    preprocess_for_analysis,
    preprocess_for_segmentation,
    segment_erythrocytes,
    extract_cell_features,
    aggregate_image_features,
    classify_image_summary,
)

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AneRBC · Diagnóstico de Anemia",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Estilos CSS personalizados
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }

    .result-card {
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 16px;
        font-family: 'Segoe UI', sans-serif;
    }
    .card-anemic {
        background: linear-gradient(135deg, #3b1a1a, #5c1e1e);
        border-left: 6px solid #e74c3c;
    }
    .card-healthy {
        background: linear-gradient(135deg, #1a3b2a, #1e5c3a);
        border-left: 6px solid #2ecc71;
    }
    .result-title { font-size: 2rem; font-weight: 800; margin-bottom: 4px; }
    .result-sub   { font-size: 1rem; opacity: 0.85; }

    .metric-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-label { color: #aaa; font-size: 0.88rem; }
    .metric-value { color: #fff; font-size: 1.05rem; font-weight: 700; }

    .aviso-medico {
        background: #1e2030;
        border: 1px solid #3a3a5c;
        border-radius: 8px;
        padding: 12px 18px;
        color: #aab0cc;
        font-size: 0.82rem;
        margin-top: 16px;
    }

    .stFileUploader label { color: #ccc !important; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #dde;
        border-bottom: 1px solid #333;
        padding-bottom: 6px;
        margin: 20px 0 12px;
    }

    /* Forzar fondo oscuro en plotly */
    .js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Carga del modelo de referencia
# ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "reference_model.pkl"

@st.cache_resource(show_spinner="Cargando modelo de referencia…")
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if not MODEL_PATH.exists():
    st.error(
        "⚠️ No se encontró `reference_model.pkl`. "
        "Ejecutá primero la celda **'Guardar modelo'** del notebook para generarlo."
    )
    st.stop()

reference_model = load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# Barra lateral
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Red_blood_cells.jpg/320px-Red_blood_cells.jpg",
        caption="Eritrocitos — frotis sanguíneo",
        use_container_width=True,
    )
    st.markdown("---")
    st.markdown("### 🔬 AneRBC · Grupo 7")
    st.markdown(
        "Detección automática de anemia a partir de imágenes de frotis sanguíneo "
        "mediante segmentación de eritrocitos y análisis morfológico."
    )
    st.markdown("---")

    st.markdown("**Umbral de clasificación**")
    threshold = st.slider(
        "Score anémico mínimo", 0.40, 0.60, 0.48, 0.01,
        help="Valor por defecto del modelo: 0.48. Ajustá según sensibilidad clínica deseada."
    )

    st.markdown("---")
    st.markdown("**Parámetros de segmentación**")
    min_distance = st.slider("Distancia mínima entre células (px)", 5, 20, 10)
    sigma        = st.slider("Suavizado (sigma)", 0.5, 3.0, 1.2, 0.1)

    st.markdown("---")
    st.caption("🏥 Herramienta de apoyo diagnóstico computacional.\nNo reemplaza el criterio clínico.")

# ─────────────────────────────────────────────
# Header principal
# ─────────────────────────────────────────────
st.markdown("# 🔬 Diagnóstico de Anemia por Frotis Sanguíneo")
st.markdown("Cargá una imagen de frotis de sangre periférica para obtener el diagnóstico automático.")

# ─────────────────────────────────────────────
# Uploader + preview interactivo
# ─────────────────────────────────────────────
col_up, col_info = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "**Seleccionar imagen de frotis sanguíneo**",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="Formatos aceptados: JPG, PNG, BMP, TIF"
    )

with col_info:
    st.markdown("""
    <div class="aviso-medico">
    <b>ℹ️ Instrucciones</b><br>
    • Imagen de frotis de sangre periférica teñida (Giemsa / May-Grünwald).<br>
    • Resolución recomendada: ≥ 800×600 px.<br>
    • Evitar imágenes sobreexpuestas o con artefactos.<br>
    • Dataset entrenado: <b>AneRBC-II</b>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Vista previa interactiva (zoom + pan) apenas se carga la imagen
# ─────────────────────────────────────────────
if uploaded_file is not None:
    img_pil_preview = Image.open(uploaded_file).convert("RGB")
    img_rgb_preview = np.array(img_pil_preview)

    st.markdown('<div class="section-title">🖼️ Vista previa de la imagen cargada</div>', unsafe_allow_html=True)
    st.caption("Usá el scroll del mouse para hacer zoom, y arrastrá para recorrer la imagen.")

    # Convertir imagen a base64 para Plotly
    buf = io.BytesIO()
    img_pil_preview.save(buf, format="PNG")
    buf.seek(0)

    h, w = img_rgb_preview.shape[:2]

    fig_preview = go.Figure()
    fig_preview.add_layout_image(
        dict(
            source=Image.fromarray(img_rgb_preview),
            xref="x", yref="y",
            x=0, y=h,
            sizex=w, sizey=h,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    fig_preview.update_layout(
        xaxis=dict(range=[0, w], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
        yaxis=dict(range=[0, h], showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        dragmode="pan",
    )
    fig_preview.update_layout(
        modebar=dict(
            bgcolor="rgba(26,26,46,0.85)",
            color="#aaa",
            activecolor="#fff",
        )
    )
    st.plotly_chart(
        fig_preview,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
            "modeBarButtonsToAdd": ["resetScale2d"],
            "toImageButtonOptions": {"format": "png", "filename": "frotis_preview"},
        }
    )

# ─────────────────────────────────────────────
# Procesamiento principal
# ─────────────────────────────────────────────
if uploaded_file is not None:
    # Reabrir el archivo (puede haberse consumido el stream)
    uploaded_file.seek(0)
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(img_pil)

    with st.spinner("⚙️ Procesando imagen… (segmentación + extracción de características)"):
        result = segment_erythrocytes(img_rgb, min_distance=min_distance, sigma=sigma)
        img_analysis = preprocess_for_analysis(img_rgb)
        df_cells = extract_cell_features(
            img_analysis,
            result["labels_features"],
            result["regions_features"]
        )
        image_summary = aggregate_image_features(
            df_cells,
            rbc_count_candidates=len(result["regions_count"])
        )
        image_summary["image_height"] = img_rgb.shape[0]
        image_summary["image_width"]  = img_rgb.shape[1]

        diagnosis = classify_image_summary(image_summary, reference_model, threshold=threshold)

    IS_ANEMIC  = diagnosis["predicted_class"] == "Anemic"
    COLOR_DX   = "#e74c3c" if IS_ANEMIC else "#2ecc71"
    CARD_CLASS = "card-anemic" if IS_ANEMIC else "card-healthy"
    ICONO      = "⚠️" if IS_ANEMIC else "✅"

    # ══════════════════════════════════════════
    # BLOQUE 1 — DIAGNÓSTICO
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-title">🩺 Diagnóstico</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card {CARD_CLASS}">
        <div class="result-title" style="color:{COLOR_DX}">
            {ICONO} {"ANÉMICO" if IS_ANEMIC else "SANO"}
        </div>
        <div class="result-sub">{diagnosis['interpretation']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Métricas de diagnóstico en 4 columnas
    m1, m2, m3, m4 = st.columns(4)

    def metric_card(label, value, col):
        col.markdown(f"""
        <div class="metric-box">
            <span class="metric-label">{label}</span>
            <span class="metric-value">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    metric_card("Score anémico",   f"{diagnosis['score_anemic']:.4f}",      m1)
    metric_card("Confianza",       f"{diagnosis['confidence']:.1%}",         m2)
    metric_card("Dist. → Anémico", f"{diagnosis['distance_to_anemic']:.3f}", m3)
    metric_card("Dist. → Sano",    f"{diagnosis['distance_to_healthy']:.3f}", m4)

    # Gauge de score
    col_gauge, col_gauge_info = st.columns([1, 1])
    with col_gauge:
        fig_g, ax_g = plt.subplots(figsize=(4, 3))
        fig_g.patch.set_facecolor("#1a1a2e")
        ax_g.set_facecolor("#1a1a2e")

        theta = np.linspace(np.pi, 0, 300)
        ax_g.plot(np.cos(theta), np.sin(theta),
                  color="#444", linewidth=14, solid_capstyle="round")

        score       = diagnosis["score_anemic"]
        theta_score = np.linspace(np.pi, np.pi - score * np.pi, 300)
        ax_g.plot(np.cos(theta_score), np.sin(theta_score),
                  color=COLOR_DX, linewidth=14, solid_capstyle="round")

        ax_g.text(0, 0.20, f"{score:.3f}", ha="center", va="center",
                  fontsize=26, fontweight="bold", color=COLOR_DX)
        ax_g.text(0, -0.20, "Score anémico", ha="center", fontsize=10, color="lightgray")
        ax_g.text(-1.05, -0.12, "Sano",    ha="center", fontsize=9, color="#2ecc71")
        ax_g.text(1.05,  -0.12, "Anémico", ha="center", fontsize=9, color="#e74c3c")

        ax_g.axvline(x=np.cos(np.pi - threshold * np.pi),
                     ymin=0.5, ymax=1.0, color="white",
                     linewidth=1.5, linestyle="--", alpha=0.5)
        ax_g.text(
            np.cos(np.pi - threshold * np.pi), -0.38,
            f"Umbral\n{threshold}", ha="center", fontsize=7.5, color="#aaa"
        )

        ax_g.set_xlim(-1.4, 1.4)
        ax_g.set_ylim(-0.5, 1.2)
        ax_g.axis("off")
        ax_g.set_title("Score diagnóstico", color="white", fontsize=10, pad=4)
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()

    with col_gauge_info:
        st.markdown(f"""
        <div class="aviso-medico" style="margin-top:30px">
        <b>Interpretación del score</b><br><br>
        El score anémico representa qué tan cercano es el perfil morfológico de la imagen
        al centroide del grupo <b>Anémico</b> respecto al grupo <b>Sano</b>.<br><br>
        • Score ≥ {threshold} → <span style="color:#e74c3c;font-weight:700">Anémico</span><br>
        • Score &lt; {threshold} → <span style="color:#2ecc71;font-weight:700">Sano</span>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # BLOQUE 2 — PARÁMETROS CALCULADOS
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-title">📊 Parámetros Calculados de la Imagen</div>', unsafe_allow_html=True)

    # Sub-bloque 2a: tabla de métricas principales
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Morfología celular**")
        params_morfo = {
            "Células analizadas (features)": int(image_summary.get("n_cells_features", 0)),
            "Candidatos (conteo)":           int(image_summary.get("rbc_count_candidates", 0)),
            "Área media (px²)":              f"{image_summary.get('area_mean', np.nan):.1f}",
            "Circularidad media":            f"{image_summary.get('circularity_mean', np.nan):.4f}",
            "Excentricidad media":           f"{image_summary.get('eccentricity_mean', np.nan):.4f}",
            "Solidez media":                 f"{image_summary.get('solidity_mean', np.nan):.4f}",
            "Diámetro equiv. medio (px)":   f"{image_summary.get('equivalent_diameter_mean', np.nan):.2f}",
            "Aspecto (major/minor)":         f"{image_summary.get('aspect_ratio_mean', np.nan):.4f}",
        }
        df_morfo = pd.DataFrame(params_morfo.items(), columns=["Parámetro", "Valor"])
        st.dataframe(df_morfo, hide_index=True, use_container_width=True)

    with col_p2:
        st.markdown("**Color y textura**")
        params_color = {
            "Palidez central media":         f"{image_summary.get('central_pallor_index_mean', np.nan):.4f}",
            "Intensidad media (gris)":       f"{image_summary.get('mean_gray_mean', np.nan):.2f}",
            "Std intensidad (gris)":         f"{image_summary.get('std_gray_mean', np.nan):.2f}",
            "Media canal L (LAB)":           f"{image_summary.get('mean_L_mean', np.nan):.2f}",
            "Media canal A (LAB)":           f"{image_summary.get('mean_A_mean', np.nan):.2f}",
            "Media canal B (LAB)":           f"{image_summary.get('mean_B_lab_mean', np.nan):.2f}",
            "GLCM Contraste (media)":        f"{image_summary.get('glcm_contrast_mean', np.nan):.4f}",
            "GLCM Homogeneidad (media)":     f"{image_summary.get('glcm_homogeneity_mean', np.nan):.4f}",
            "GLCM Energía (media)":          f"{image_summary.get('glcm_energy_mean', np.nan):.4f}",
        }
        df_color = pd.DataFrame(params_color.items(), columns=["Parámetro", "Valor"])
        st.dataframe(df_color, hide_index=True, use_container_width=True)

    # Sub-bloque 2b: comparativa features vs referencia
    st.markdown('<div class="section-title">📋 Comparativa de Features vs. Referencia</div>', unsafe_allow_html=True)

    raw_means   = reference_model["raw_means"]
    feature_rows = []
    for f in reference_model["features"]:
        value        = image_summary.get(f, np.nan)
        mean_anemic  = raw_means.loc["Anemic",  f]
        mean_healthy = raw_means.loc["Healthy", f]
        if pd.isna(value):
            closer_to = "—"
        else:
            closer_to = "Anemic" if abs(value - mean_anemic) < abs(value - mean_healthy) else "Healthy"
        feature_rows.append({
            "Feature":              f,
            "Valor imagen":         round(value, 4) if not pd.isna(value) else np.nan,
            "Media Anémico (ref.)": round(mean_anemic, 4),
            "Media Sano (ref.)":    round(mean_healthy, 4),
            "Más cercano a":        closer_to,
        })

    df_feat = pd.DataFrame(feature_rows)

    def color_closer(val):
        if val == "Anemic":
            return "color: #e74c3c; font-weight: bold"
        elif val == "Healthy":
            return "color: #2ecc71; font-weight: bold"
        return ""

    st.dataframe(
        df_feat.style.applymap(color_closer, subset=["Más cercano a"]),
        hide_index=True,
        use_container_width=True,
        height=320
    )

    # Sub-bloque 2c: gráfico de barras top features
    st.markdown('<div class="section-title">📈 Top Features Discriminativas</div>', unsafe_allow_html=True)

    df_plot = df_feat.dropna(subset=["Valor imagen"]).copy()
    ref_avg = (df_plot["Media Anémico (ref.)"] + df_plot["Media Sano (ref.)"]) / 2
    ref_avg = ref_avg.replace(0, 1)
    df_plot["Valor normalizado"] = df_plot["Valor imagen"] / ref_avg

    n_top  = min(12, len(df_plot))
    df_top = df_plot.head(n_top).reset_index(drop=True)

    fig_bar, ax_bar = plt.subplots(figsize=(8, n_top * 0.55 + 1))
    fig_bar.patch.set_facecolor("#1a1a2e")
    ax_bar.set_facecolor("#1a1a2e")

    bar_colors = ["#e74c3c" if c == "Anemic" else "#2ecc71"
                  for c in df_top["Más cercano a"]]
    y_pos = np.arange(len(df_top))
    ax_bar.barh(y_pos, df_top["Valor normalizado"], color=bar_colors, alpha=0.80, height=0.6)
    ax_bar.axvline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f[:28] for f in df_top["Feature"]], fontsize=8, color="lightgray")
    ax_bar.set_xlabel("Valor imagen / media referencia", color="lightgray", fontsize=9)
    ax_bar.tick_params(colors="lightgray", labelsize=8)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")
    ax_bar.set_title(
        "Valor normalizado respecto a media de referencia\n(rojo = patrón anémico · verde = patrón sano)",
        color="white", fontsize=9, pad=6
    )
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

    # Sub-bloque 2d: histograma de circularidad
    if not df_cells.empty and "circularity" in df_cells.columns:
        with st.expander("📉 Distribución de circularidad celular", expanded=False):
            fig_hist, ax_hist = plt.subplots(figsize=(6, 3))
            fig_hist.patch.set_facecolor("#1a1a2e")
            ax_hist.set_facecolor("#1a1a2e")
            ax_hist.hist(df_cells["circularity"].dropna(), bins=20, color=COLOR_DX, alpha=0.8, edgecolor="#111")
            ax_hist.axvline(df_cells["circularity"].mean(), color="white", linestyle="--", linewidth=1.2, label="Media")
            ax_hist.set_xlabel("Circularidad", color="lightgray")
            ax_hist.set_ylabel("N° de células", color="lightgray")
            ax_hist.tick_params(colors="lightgray")
            ax_hist.legend(fontsize=8, labelcolor="white", facecolor="#222")
            for spine in ax_hist.spines.values():
                spine.set_edgecolor("#333")
            ax_hist.set_title("Distribución de circularidad por célula detectada", color="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_hist)
            plt.close()

    # ══════════════════════════════════════════
    # BLOQUE 3 — IMÁGENES DE SEGMENTACIÓN
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-title">🔬 Segmentación de Eritrocitos</div>', unsafe_allow_html=True)

    # Fila 1: imagen original + bboxes | máscara binaria | watershed coloreado
    fig_seg, axes_seg = plt.subplots(1, 3, figsize=(15, 4.5))
    fig_seg.patch.set_facecolor("#0f1117")

    ax = axes_seg[0]
    ax.imshow(img_rgb)
    for region in result["regions_features"]:
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            linewidth=0.9, edgecolor=COLOR_DX, facecolor="none"
        )
        ax.add_patch(rect)
    n_det = len(result["regions_features"])
    ax.set_title(f"Imagen original\n{n_det} eritrocitos detectados", color="white", fontsize=10, pad=6)
    ax.axis("off")

    ax = axes_seg[1]
    ax.imshow(result["binary"], cmap="gray")
    ax.set_title("Máscara binaria\n(Otsu + morfología)", color="white", fontsize=10, pad=6)
    ax.axis("off")

    ax = axes_seg[2]
    ax.imshow(result["labels_features"], cmap="nipy_spectral",
              vmin=0, vmax=max(result["labels_features"].max(), 1))
    ax.set_title("Segmentación Watershed\n(etiquetado por célula)", color="white", fontsize=10, pad=6)
    ax.axis("off")

    for a in axes_seg:
        a.set_facecolor("#1a1a2e")
    plt.tight_layout()
    st.pyplot(fig_seg)
    plt.close()

    # Fila 2: imagen pre-procesada | mapa de distancia | marcadores
    st.markdown("**Etapas internas del pipeline**")

    fig_pipe, axes_pipe = plt.subplots(1, 3, figsize=(15, 4.5))
    fig_pipe.patch.set_facecolor("#0f1117")

    ax = axes_pipe[0]
    ax.imshow(result["enhanced"])
    ax.set_title("Pre-procesamiento\n(CLAHE sobre canal L)", color="white", fontsize=10, pad=6)
    ax.axis("off")

    ax = axes_pipe[1]
    im = ax.imshow(result["dist"], cmap="hot")
    ax.set_title("Transformada de distancia\n(entrada Watershed)", color="white", fontsize=10, pad=6)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    ax = axes_pipe[2]
    overlay = img_rgb.copy()
    labels_count = result["labels_count"]
    for region in result["regions_count"]:
        cy, cx = region.centroid
        cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 80, 80), -1)
    ax.imshow(overlay)
    n_count = len(result["regions_count"])
    ax.set_title(f"Centroides detectados (conteo)\n{n_count} candidatos", color="white", fontsize=10, pad=6)
    ax.axis("off")

    for a in axes_pipe:
        a.set_facecolor("#1a1a2e")
    plt.tight_layout()
    st.pyplot(fig_pipe)
    plt.close()

    # Imagen aumentada con zoom/pan (segmentación interactiva)
    with st.expander("🔍 Vista interactiva de la segmentación (zoom + pan)", expanded=False):
        st.caption("Scroll para zoom · Arrastrá para recorrer · Doble click para resetear")

        # Crear imagen con bboxes de colores
        seg_img = img_rgb.copy()
        for region in result["regions_features"]:
            minr, minc, maxr, maxc = region.bbox
            color_cv2 = (231, 76, 60) if IS_ANEMIC else (46, 204, 113)
            cv2.rectangle(seg_img, (minc, minr), (maxc, maxr), color_cv2, 2)

        h2, w2 = seg_img.shape[:2]
        fig_zoom = go.Figure()
        fig_zoom.add_layout_image(
            dict(
                source=Image.fromarray(seg_img),
                xref="x", yref="y",
                x=0, y=h2,
                sizex=w2, sizey=h2,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )
        fig_zoom.update_layout(
            xaxis=dict(range=[0, w2], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
            yaxis=dict(range=[0, h2], showgrid=False, zeroline=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            dragmode="pan",
        )
        st.plotly_chart(
            fig_zoom,
            use_container_width=True,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                "toImageButtonOptions": {"format": "png", "filename": "segmentacion"},
            }
        )

    # ══════════════════════════════════════════
    # BLOQUE 4 — EXPORTAR
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown('<div class="section-title">💾 Exportar resultados</div>', unsafe_allow_html=True)

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        diagnosis_export  = {**image_summary, **diagnosis}
        df_diag_export    = pd.DataFrame([diagnosis_export])
        csv_diag          = df_diag_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar diagnóstico (CSV)",
            data=csv_diag,
            file_name="diagnostico.csv",
            mime="text/csv"
        )

    with col_d2:
        csv_feat = df_feat.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar features vs. referencia (CSV)",
            data=csv_feat,
            file_name="features_diagnostico.csv",
            mime="text/csv"
        )

    # Aviso médico
    st.markdown("""
    <div class="aviso-medico">
    ⚕️ <b>AVISO IMPORTANTE:</b> Este sistema es una herramienta de apoyo computacional basada en análisis de imagen.
    Los resultados no constituyen un diagnóstico clínico definitivo y no reemplazan la evaluación médica profesional
    ni los análisis de laboratorio hematológico. Toda decisión clínica debe ser tomada por un profesional de la salud habilitado.
    </div>
    """, unsafe_allow_html=True)

else:
    # Estado vacío: instrucciones
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🖼️", "Cargar imagen",           "Subí una imagen JPG/PNG de frotis sanguíneo teñido."),
        (c2, "⚙️", "Procesamiento automático", "El sistema segmenta eritrocitos y extrae características morfológicas."),
        (c3, "📋", "Resultado diagnóstico",    "Obtené el diagnóstico con score, confianza y comparativa de features."),
    ]:
        col.markdown(f"""
        <div style="background:#1a1a2e;border-radius:12px;padding:24px;text-align:center;">
            <div style="font-size:2.2rem">{icon}</div>
            <div style="color:#dde;font-weight:700;margin:8px 0">{title}</div>
            <div style="color:#aab;font-size:0.88rem">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
