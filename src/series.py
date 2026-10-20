PLANES = ["Axial", "Coronal", "Sagittal"]


def select_series_per_plane(study_uid, series_df):
    """Pick one series per plane, preferring Fluid_Sensitive=1 (fluid, cartilage, ligament detail)."""
    study = series_df[series_df["StudyInstanceUID"] == study_uid]
    selected = {}
    for plane in PLANES:
        candidates = study[study["Anatomical_Plane"] == plane]
        fluid = candidates[candidates["Fluid_Sensitive"] == 1]
        pool = fluid if len(fluid) else candidates  # fallback when no fluid-sensitive series exists
        selected[plane] = pool.iloc[0]["SeriesInstanceUID"]
    return selected