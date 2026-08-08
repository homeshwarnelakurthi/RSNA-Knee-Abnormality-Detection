"""EDA #1: label availability, prevalence, co-occurrence, and report characteristics."""
import pandas as pd, numpy as np, re, collections, json, sys

D = r"H:\RSNA Knee Abnormality Detection\data"
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

tr = pd.read_csv(f"{D}/train.csv")
ts = pd.read_csv(f"{D}/train_series.csv")

print("="*70); print("TRAIN.CSV SHAPE:", tr.shape); print("COLUMNS:", list(tr.columns))
print("\nDTYPES:\n", tr.dtypes)

# ---- label availability -------------------------------------------------
lab = tr[LABELS]
n_lab_present = lab.notna().sum(axis=1)
print("\n" + "="*70)
print("LABEL AVAILABILITY (how many of the 12 labels are non-null per study)")
print(n_lab_present.value_counts().sort_index().to_string())
fully = (n_lab_present == 12).sum()
none_ = (n_lab_present == 0).sum()
print(f"\nStudies with ALL 12 labels : {fully} ({fully/len(tr):.1%})")
print(f"Studies with NO labels     : {none_} ({none_/len(tr):.1%})")
print(f"Total studies              : {len(tr)}")

# ---- report availability ------------------------------------------------
has_rep = tr["Report"].notna() & (tr["Report"].astype(str).str.strip() != "")
print(f"\nStudies with a Report      : {has_rep.sum()} ({has_rep.mean():.1%})")
print("\nCross-tab: has_report x has_labels")
print(pd.crosstab(has_rep.rename("has_report"), (n_lab_present==12).rename("fully_labeled")))

# ---- prevalence ---------------------------------------------------------
print("\n" + "="*70); print("PREVALENCE among labeled studies")
lab_sub = lab[n_lab_present == 12]
prev = pd.DataFrame({
    "n_pos": lab_sub.sum().astype(int),
    "prevalence": lab_sub.mean(),
}).sort_values("prevalence", ascending=False)
print(prev.to_string())
print(f"\nMean labels positive per study: {lab_sub.sum(axis=1).mean():.2f}")
print("Distribution of #positives per study:")
print(lab_sub.sum(axis=1).value_counts().sort_index().to_string())

# ---- co-occurrence ------------------------------------------------------
print("\n" + "="*70); print("LABEL CORRELATION (phi)")
print(lab_sub.astype(float).corr().round(2).to_string())

# ---- sex ----------------------------------------------------------------
print("\n" + "="*70)
if "PatientSex" in tr.columns:
    print("PatientSex:"); print(tr["PatientSex"].value_counts(dropna=False).to_string())
else:
    print("!! PatientSex column ABSENT from train.csv (data description claims it exists) !!")

# ---- reports: language / length ----------------------------------------
rep = tr.loc[has_rep, "Report"].astype(str)
print("\n" + "="*70); print("REPORT LENGTH (chars)")
print(rep.str.len().describe().round(1).to_string())
print("\nREPORT LENGTH (words)")
print(rep.str.split().str.len().describe().round(1).to_string())

# crude script/charset detection
def script_of(s):
    if re.search(r"[\u4e00-\u9fff]", s): return "CJK-han"
    if re.search(r"[\u3040-\u30ff]", s): return "Japanese-kana"
    if re.search(r"[\uac00-\ud7af]", s): return "Korean"
    if re.search(r"[\u0400-\u04ff]", s): return "Cyrillic"
    if re.search(r"[\u0600-\u06ff]", s): return "Arabic"
    if re.search(r"[\u0590-\u05ff]", s): return "Hebrew"
    if re.search(r"[\u0e00-\u0e7f]", s): return "Thai"
    if re.search(r"[\u0900-\u097f]", s): return "Devanagari"
    if re.search(r"[àâçéèêëîïôùûüÿœæ]", s, re.I): return "Latin-accented(fr-ish)"
    if re.search(r"[äöüß]", s, re.I): return "Latin-accented(de-ish)"
    if re.search(r"[áéíóúñ¿¡]", s, re.I): return "Latin-accented(es/pt-ish)"
    return "Latin-plain"
print("\nCRUDE SCRIPT DETECTION:")
print(rep.map(script_of).value_counts().to_string())

# most common tokens (gives a language fingerprint)
toks = collections.Counter()
for s in rep.sample(min(2000, len(rep)), random_state=0):
    toks.update(re.findall(r"[^\W\d_]{3,}", s.lower(), re.UNICODE))
print("\nTOP 60 TOKENS across reports:")
print(", ".join(f"{w}({c})" for w, c in toks.most_common(60)))

# ---- series structure ---------------------------------------------------
print("\n" + "="*70); print("TRAIN_SERIES.CSV SHAPE:", ts.shape)
print("\nSeries per study:")
spc = ts.groupby("StudyInstanceUID").size()
print(spc.describe().round(2).to_string())
print("\nSeries-per-study value counts:"); print(spc.value_counts().sort_index().head(15).to_string())
print("\nAnatomical_Plane:"); print(ts["Anatomical_Plane"].value_counts(dropna=False).to_string())
print("\nFluid_Sensitive x Fat_Suppression:")
print(pd.crosstab(ts["Fluid_Sensitive"], ts["Fat_Suppression"]).to_string())
print("\nPlane x (Fluid,FatSup) combos:")
ts["combo"] = ts["Anatomical_Plane"].astype(str)+"|FS"+ts["Fluid_Sensitive"].astype(str)+"|FatSat"+ts["Fat_Suppression"].astype(str)
print(ts["combo"].value_counts().to_string())

# how many studies have each canonical combo?
print("\nStudy coverage per combo (fraction of studies having >=1 such series):")
nstud = ts["StudyInstanceUID"].nunique()
cov = ts.groupby("combo")["StudyInstanceUID"].nunique().sort_values(ascending=False)/nstud
print(cov.round(3).to_string())
print(f"\nStudies in train_series: {nstud}  |  studies in train.csv: {len(tr)}")
print("Studies in train.csv missing from train_series:", len(set(tr.StudyInstanceUID)-set(ts.StudyInstanceUID)))
