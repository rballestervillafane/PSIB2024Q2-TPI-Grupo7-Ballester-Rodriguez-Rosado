"""
pipeline.py — Funciones de procesamiento extraídas del notebook TPI_AneRBC_Grupo7
Grupo 7 · Ballester Villafañe · Rosado · Rodriguez Torcelli
"""

import numpy as np
import pandas as pd
import cv2

from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.morphology import disk
from skimage.segmentation import watershed, clear_border


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRE-PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_analysis(img_rgb):
    """
    Pre-procesamiento orientado al análisis de características:
    - Suavizado Gaussiano ligero para reducir ruido de alta frecuencia.
    - CLAHE sobre canal L (espacio LAB) para mejorar contraste local.
    """
    img = cv2.GaussianBlur(img_rgb.copy(), (5, 5), sigmaX=1)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_for_segmentation(img_rgb):
    """
    Pre-procesamiento orientado a la segmentación.
    Retorna imagen mejorada (RGB) y mapa de grises filtrado.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    L = clahe.apply(L)
    lab_enhanced = cv2.merge([L, A, B])
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return img_enhanced, gray


# ─────────────────────────────────────────────────────────────────────────────
# 5. SEGMENTACIÓN DE ERITROCITOS
# ─────────────────────────────────────────────────────────────────────────────

def get_rbc_binary_mask(gray, min_area=120, close_radius=2, remove_border=False):
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh
    binary = morphology.binary_closing(binary, footprint=disk(close_radius))
    binary = ndi.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.remove_small_holes(binary, area_threshold=min_area // 2)
    if remove_border:
        binary = clear_border(binary)
    return binary


def split_touching_cells(binary, min_distance=10, sigma=1.2):
    dist = ndi.distance_transform_edt(binary)
    dist_smooth = ndi.gaussian_filter(dist, sigma=sigma)

    coords = peak_local_max(
        dist_smooth,
        min_distance=min_distance,
        labels=binary,
        exclude_border=False
    )

    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    markers, _ = ndi.label(markers > 0)

    labels_ws = watershed(-dist_smooth, markers, mask=binary)
    return labels_ws, dist_smooth, markers


def filter_regions(labels,
                   min_area=80,
                   max_area=3000,
                   min_circularity=0.20,
                   min_solidity=0.60,
                   max_eccentricity=0.99):
    filtered  = np.zeros_like(labels, dtype=np.int32)
    new_label = 1

    for region in measure.regionprops(labels):
        area      = region.area
        perimeter = region.perimeter
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        if circularity < min_circularity:
            continue
        if region.solidity < min_solidity:
            continue
        if region.eccentricity > max_eccentricity:
            continue

        filtered[labels == region.label] = new_label
        new_label += 1

    regions = measure.regionprops(filtered)
    return filtered, regions


def segment_erythrocytes(img_rgb,
                          min_area_mask=80,
                          close_radius=2,
                          min_distance=10,
                          sigma=1.2,
                          remove_border=False):
    """
    Pipeline de segmentación de eritrocitos.
    Acepta parámetros ajustables desde la UI.
    """
    img_enhanced, gray = preprocess_for_segmentation(img_rgb)

    binary = get_rbc_binary_mask(
        gray,
        min_area=min_area_mask,
        close_radius=close_radius,
        remove_border=remove_border
    )

    labels_ws, dist, markers = split_touching_cells(
        binary, min_distance=min_distance, sigma=sigma
    )

    labels_count, regions_count = filter_regions(
        labels_ws,
        min_area=100,
        max_area=5000,
        min_circularity=0.20,
        min_solidity=0.55,
        max_eccentricity=0.99
    )

    labels_features, regions_features = filter_regions(
        labels_ws,
        min_area=150,
        max_area=3500,
        min_circularity=0.45,
        min_solidity=0.75,
        max_eccentricity=0.97
    )

    return {
        "enhanced"         : img_enhanced,
        "gray"             : gray,
        "binary"           : binary,
        "dist"             : dist,
        "markers"          : markers,
        "labels_ws"        : labels_ws,
        "labels_count"     : labels_count,
        "regions_count"    : regions_count,
        "labels_features"  : labels_features,
        "regions_features" : regions_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXTRACCIÓN DE CARACTERÍSTICAS POR CÉLULA
# ─────────────────────────────────────────────────────────────────────────────

def compute_circularity(area, perimeter):
    if perimeter == 0:
        return np.nan
    return 4 * np.pi * area / (perimeter ** 2)


def compute_central_pallor(gray_image, region_mask):
    region_mask = region_mask.astype(bool)
    if region_mask.sum() == 0:
        return np.nan

    central_mask = morphology.erosion(region_mask, disk(3))
    if central_mask.sum() == 0:
        return np.nan

    border_mask = region_mask & (~central_mask)
    if border_mask.sum() == 0:
        return np.nan

    central_mean = np.mean(gray_image[central_mask])
    border_mean  = np.mean(gray_image[border_mask])

    if border_mean == 0:
        return np.nan

    return central_mean / border_mean


def compute_glcm_features(gray_patch):
    if gray_patch.size == 0:
        return np.nan, np.nan, np.nan
    patch = gray_patch.copy()
    if patch.max() == patch.min():
        return 0.0, 1.0, 1.0

    patch_norm = ((patch - patch.min()) / (patch.max() - patch.min()) * 7).astype(np.uint8)
    glcm = graycomatrix(
        patch_norm,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )
    contrast    = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy      = graycoprops(glcm, "energy")[0, 0]
    return contrast, homogeneity, energy


def extract_cell_features(img_rgb_analysis, labels, regions):
    features = []
    gray = cv2.cvtColor(img_rgb_analysis, cv2.COLOR_RGB2GRAY)
    lab  = cv2.cvtColor(img_rgb_analysis, cv2.COLOR_RGB2LAB)

    for region in regions:
        label_id  = region.label
        cell_mask = labels == label_id

        area        = region.area
        perimeter   = region.perimeter
        circularity = compute_circularity(area, perimeter)
        aspect_ratio = (
            region.major_axis_length / region.minor_axis_length
            if region.minor_axis_length > 0 else np.nan
        )

        minr, minc, maxr, maxc = region.bbox
        gray_patch = gray[minr:maxr, minc:maxc]

        gray_vals = gray[cell_mask]
        L_vals    = lab[:, :, 0][cell_mask]
        A_vals    = lab[:, :, 1][cell_mask]
        B_vals    = lab[:, :, 2][cell_mask]
        R_vals    = img_rgb_analysis[:, :, 0][cell_mask]
        G_vals    = img_rgb_analysis[:, :, 1][cell_mask]
        Br_vals   = img_rgb_analysis[:, :, 2][cell_mask]

        pallor_index        = compute_central_pallor(gray, cell_mask)
        glcm_c, glcm_h, glcm_e = compute_glcm_features(gray_patch)

        features.append({
            "label"               : label_id,
            "area"                : area,
            "perimeter"           : perimeter,
            "circularity"         : circularity,
            "eccentricity"        : region.eccentricity,
            "solidity"            : region.solidity,
            "extent"              : region.extent,
            "orientation"         : region.orientation,
            "equivalent_diameter" : region.equivalent_diameter,
            "major_axis_length"   : region.major_axis_length,
            "minor_axis_length"   : region.minor_axis_length,
            "aspect_ratio"        : aspect_ratio,
            "mean_gray"           : np.mean(gray_vals),
            "std_gray"            : np.std(gray_vals),
            "mean_L"              : np.mean(L_vals),
            "mean_A"              : np.mean(A_vals),
            "mean_B_lab"          : np.mean(B_vals),
            "mean_R"              : np.mean(R_vals),
            "mean_G"              : np.mean(G_vals),
            "mean_B_rgb"          : np.mean(Br_vals),
            "central_pallor_index": pallor_index,
            "glcm_contrast"       : glcm_c,
            "glcm_homogeneity"    : glcm_h,
            "glcm_energy"         : glcm_e,
        })

    return pd.DataFrame(features)


# ─────────────────────────────────────────────────────────────────────────────
# 7. AGREGACIÓN POR IMAGEN
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_image_features(df_cells, rbc_count_candidates):
    """Agrega características celulares a nivel de imagen (media, std, percentiles)."""
    if df_cells.empty:
        return {
            "n_cells_features"     : 0,
            "rbc_count_candidates" : rbc_count_candidates,
        }

    numeric_cols = df_cells.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"label"}
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    summary = {
        "n_cells_features"    : len(df_cells),
        "rbc_count_candidates": rbc_count_candidates,
    }

    for col in feature_cols:
        vals = df_cells[col].dropna()
        if len(vals) == 0:
            summary[f"{col}_mean"] = np.nan
            summary[f"{col}_std"]  = np.nan
            summary[f"{col}_p25"]  = np.nan
            summary[f"{col}_p75"]  = np.nan
        else:
            summary[f"{col}_mean"] = vals.mean()
            summary[f"{col}_std"]  = vals.std()
            summary[f"{col}_p25"]  = vals.quantile(0.25)
            summary[f"{col}_p75"]  = vals.quantile(0.75)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def classify_image_summary(image_summary, reference_model, threshold=0.48):
    """
    Clasifica una imagen usando el modelo de referencia por centroides normalizados.
    El umbral es ajustable desde la interfaz.
    """
    features    = reference_model["features"]
    global_mean = reference_model["global_mean"]
    global_std  = reference_model["global_std"]
    centroids   = reference_model["centroids"]

    values = [
        image_summary.get(f, np.nan)
        if not pd.isna(image_summary.get(f, np.nan))
        else global_mean[f]
        for f in features
    ]
    x   = pd.Series(values, index=features)
    x_z = (x - global_mean[features]) / global_std[features]

    distances = {
        cls: np.sqrt(np.sum((x_z - centroids.loc[cls]) ** 2))
        for cls in centroids.index
    }

    d_anemic  = distances["Anemic"]
    d_healthy = distances["Healthy"]
    total     = d_anemic + d_healthy

    score_anemic = d_healthy / total if total > 0 else 0.5

    if score_anemic >= threshold:
        predicted_class = "Anemic"
        interpretation  = "Compatible con patrón anémico"
        confidence      = score_anemic
    else:
        predicted_class = "Healthy"
        interpretation  = "Compatible con patrón sano"
        confidence      = 1 - score_anemic

    return {
        "predicted_class"     : predicted_class,
        "interpretation"      : interpretation,
        "score_anemic"        : round(score_anemic, 4),
        "confidence"          : round(confidence, 4),
        "distance_to_anemic"  : round(d_anemic, 4),
        "distance_to_healthy" : round(d_healthy, 4),
    }
