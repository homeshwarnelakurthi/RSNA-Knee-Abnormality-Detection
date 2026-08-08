"""
Job B (B1-B3) — report statistics. Phase 0 research.

Aggregate counts ONLY. No report text is printed or written to disk: report text is
Competition Data and must not leave this machine.

B1  Does severity language actually appear often enough for the graded-target thesis?
B2  Are reports structured or free-form, and does that track language/site?
B3  Does report length vary in a way that changes what "not mentioned" means?
"""
import re, unicodedata
import numpy as np
import pandas as pd

D = r"H:\RSNA Knee Abnormality Detection\data"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

tr = pd.read_csv(f"{D}/train.csv")
lang = pd.read_csv(r"H:\RSNA Knee Abnormality Detection\eda\report_lang.csv")
tr = tr.merge(lang, on="StudyInstanceUID", how="left")

_PRE = str.maketrans({"ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d"})


def norm(t):
    if not isinstance(t, str):
        return ""
    t = t.translate(_PRE).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[ \t]+", " ", t)


tr["N"] = tr["Report"].map(norm)

# --------------------------------------------------------------- lexicons
def rx(*a):
    return re.compile("|".join(a))

SEV = {
    "minimal": rx(r"\btrace\b", r"\bminimal", r"\bminim", r"\bminiem", r"\bspur\b",
                  r"\bαμελητε", r"ελαχιστ", r"минимал", r"\baz miktarda\b", r"\bçok az\b"),
    "mild":    rx(r"\bmild", r"\bslight", r"\bsmall\b", r"\bleve\b", r"\bligera", r"\bpequen",
                  r"\bleger", r"\bfaible\b", r"\bpetit", r"\blicht", r"\bgering", r"\bklein",
                  r"\bhafif", r"\bkucuk\b", r"ηπι", r"\bμικρ", r"\bлек", r"\bмалк", r"\bblag",
                  r"\bmali\b", r"\bmanji\b", r"\bdiskret", r"\bdiscret"),
    "moderate": rx(r"\bmoderate", r"\bmoderad", r"\bmodere", r"\bmatig", r"\bmassig",
                   r"\bmittelgradig", r"\borta\b", r"μετρι", r"\bумерен", r"\bсреден",
                   r"\bumjeren", r"\bsrednj"),
    "severe":  rx(r"\bsevere", r"\bmarked", r"\blarge\b", r"\bmassive", r"\bgross\b",
                  r"\badvanced", r"\bextensive", r"\bhigh.grade", r"\bfull.thickness",
                  r"\bcomplete\b", r"\bsevero", r"\bgrave", r"\bimportante", r"\bgrande",
                  r"\bextens", r"\bavanzad", r"\bernstig", r"\bgroot", r"\buitgebreid",
                  r"\bausgepragt", r"\bschwer", r"\bstark", r"\bhochgradig", r"\bgross",
                  r"\bileri\b", r"\bbelirgin", r"\bbuyuk\b", r"\byaygin", r"\bciddi",
                  r"σοβαρ", r"μεγαλ", r"εκτεταμεν", r"\bтежк", r"\bизразен", r"\bголям",
                  r"\btezak", r"\bizrazit", r"\bvelik", r"\bopsezn"),
}
GRADE = rx(r"\bgrade?\s*[0-4iv]", r"\bgrado\s*[0-4iv]", r"\bgraad\s*[0-4iv]",
           r"\bgrad\s*[0-4iv]", r"\bderece\s*[0-4]", r"\bevre\s*[0-4]",
           r"βαθμ\w*\s*[0-4ivα-δ]", r"степен\w*\s*[0-4]", r"\b[0-4]\s*степен",
           r"outerbridge", r"\bicrs\b", r"kellgren", r"\bstadi\w*\s*[0-4iv]",
           r"\b[0-4]\s*\.?\s*grad", r"\bgr\.?\s*[1-4]\b")
ANY_SEV = rx(*[p.pattern for p in SEV.values()], GRADE.pattern)

# Finding cues per label. Deliberately broad: B1 asks how often severity language
# accompanies a mention, not whether the mention is positive.
CUE = {
    "ACL": rx(r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blca\b",
              r"croise anterieur", r"voorste kruisband", r"\bvkb\b", r"vordere[sn]? kreuzband",
              r"on capraz", r"prednj\w* krizn", r"χιαστ", r"кръстна", r"крестообраз"),
    "MCL": rx(r"medial collateral", r"\bmcl\b", r"tibial collateral", r"colateral (medial|interno)",
              r"collateral (medial|interne)", r"mediale collaterale", r"binnenband", r"innenband",
              r"ic yan bag", r"medial kollateral", r"medijaln\w* kolateraln", r"εσω πλαγι",
              r"медиален колатерал", r"вътрешна странична"),
    "Medial Meniscus": rx(r"medial menisc", r"menisco (medial|interno)", r"menisque (medial|interne)",
                          r"mediale meniscus", r"binnenmeniscus", r"innenmeniskus", r"mediale[rn]? meniskus",
                          r"medyal menisk", r"\bic menisk", r"medijaln\w* meniskus", r"εσω μηνισκ",
                          r"медиалн\w* менискус", r"вътрешния менискус"),
    "Lateral Meniscus": rx(r"lateral menisc", r"menisco (lateral|externo)", r"menisque (lateral|externe)",
                           r"laterale meniscus", r"buitenmeniscus", r"aussenmeniskus", r"laterale[rn]? meniskus",
                           r"lateral menisk", r"\bdis menisk", r"lateraln\w* meniskus", r"εξω μηνισκ",
                           r"латералн\w* менискус", r"външния менискус"),
    "Medial OA": rx(r"medial (femorotibial|tibiofemoral|compartment)", r"femorotibial (medial|interno)",
                    r"medial (femoral|tibial) (condyle|plateau)", r"medialen kompartiment",
                    r"medyal (femorotibial|kompartman)", r"medijaln\w* (femorotib|kompartm)",
                    r"εσω διαμερισμα", r"медиалн\w* (компартм|отдел)"),
    "Lateral OA": rx(r"lateral (femorotibial|tibiofemoral|compartment)", r"femorotibial (lateral|externo)",
                     r"lateral (femoral|tibial) (condyle|plateau)", r"lateralen kompartiment",
                     r"lateral (femorotibial|kompartman)", r"lateraln\w* (femorotib|kompartm)",
                     r"εξω διαμερισμα", r"латералн\w* (компартм|отдел)"),
    "PF OA": rx(r"patellofemoral", r"femoropatell", r"patelofemoral", r"retropatellar", r"retrorotulian",
                r"trochlea", r"troclea", r"troklea", r"\bpatell", r"\brotulian", r"\brotula\b",
                r"επιγονατιδ", r"τροχιλ", r"пател"),
    "Effusion": rx(r"\beffusion", r"joint fluid", r"derrame", r"liquido articular", r"epanchement",
                   r"gewrichtsvocht", r"gelenkergu", r"\bergu(ss|ß)", r"eklem\w* ic\w* sivi",
                   r"efuzyon", r"sivi (artis|miktari)", r"izljev", r"zglobn\w* tekucin",
                   r"αρθρικ\w* υγρ", r"ενδαρθρικ", r"излив", r"ставна течност"),
    "Synovitis": rx(r"synovit", r"sinovit", r"synovial", r"sinovij", r"synovialitis",
                    r"υμενιτιδ", r"συνοβιτ", r"аρθρικου υμεν", r"синовит", r"pannus"),
    "Baker's": rx(r"baker", r"popliteal cyst", r"quiste popliteo", r"kyste poplite",
                  r"popliteale? cyst", r"poplitealzyste", r"popliteal kist", r"bakerova",
                  r"poplitealn\w* cist", r"κυστη baker", r"бейкер", r"поплитеал"),
    "Contusion": rx(r"contusion", r"bone bruise", r"bone marrow edema", r"marrow oedema",
                    r"edema oseo", r"contusion osea", r"oedeme osseux", r"botoedeem",
                    r"knochenmarksodem", r"knochenodem", r"kemik ilig\w* odem", r"kontuzyon",
                    r"kostn\w* edem", r"οστικο οιδημα", r"костномозъчен едем", r"костен едем"),
    "Fracture": rx(r"fractur", r"fraktur", r"fractura", r"breuk\b", r"kirik", r"prijelom",
                   r"καταγμα", r"фрактур", r"счупв"),
}

# clause splitter
SPLIT = re.compile(r"(?<=[.;!?:])\s+|\n+")

print("=" * 78)
print("B3  REPORT LENGTH AND COMPLETENESS")
print("=" * 78)
tr["nwords"] = tr["N"].str.split().str.len()
print("\nWords per report, overall:")
print(tr["nwords"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
print("\nWords per report, by language:")
print(tr.groupby("lang")["nwords"].describe(percentiles=[.5]).round(1)
        [["count", "mean", "50%", "min", "max"]].sort_values("count", ascending=False).to_string())
print("\nShare of reports under 50 words (likely omit negatives), by language:")
short = tr.assign(s=tr["nwords"] < 50).groupby("lang")["s"].mean().sort_values(ascending=False)
print((short * 100).round(1).to_string())

print("\n" + "=" * 78)
print("B2  STRUCTURE: do reports use section headers?")
print("=" * 78)
HEAD = rx(r"\bfindings?\s*:", r"\bimpression\s*:", r"\bconclusion", r"\btechnique\s*:",
          r"\bhallazgos\s*:", r"\bimpresion\s*:", r"\bconstatations", r"\bbevindingen",
          r"\bbulgular\s*:", r"\bsonuc\s*:", r"\bευρηματα", r"\bfindings\b",
          r"\bmedial compartment\s*:", r"\blateral compartment\s*:", r"\bнаходка",
          r"\bbefund", r"\bbeurteilung", r"\bnalaz", r"\bzakljucak")
tr["structured"] = tr["N"].str.contains(HEAD)
print(f"\nReports with recognisable section headers: {tr['structured'].mean():.1%}")
print("\nBy language:")
st = tr.groupby("lang").agg(n=("structured", "size"), structured=("structured", "mean"))
print(st.assign(structured=(st["structured"] * 100).round(1)).sort_values("n", ascending=False).to_string())
print("\n-> if structure correlates strongly with language, a labeler tuned on one")
print("   format acquires a site-correlated bias. Grouped CV will expose it late.")

print("\n" + "=" * 78)
print("B1  *** SEVERITY LANGUAGE — THE KEY NUMBER FOR THE GRADED-TARGET THESIS ***")
print("=" * 78)

print("\nReports containing ANY severity/grade cue anywhere:")
tr["has_sev"] = tr["N"].str.contains(ANY_SEV)
print(f"  {tr['has_sev'].mean():.1%}  ({tr['has_sev'].sum()} of {len(tr)})")
print("\nBy severity tier (share of all reports containing >=1 such word):")
for k, p in SEV.items():
    print(f"  {k:<9} {tr['N'].str.contains(p).mean():6.1%}")
print(f"  {'grade':<9} {tr['N'].str.contains(GRADE).mean():6.1%}")

print("\nBy language (share of reports with any severity cue):")
bl = tr.groupby("lang").agg(n=("has_sev", "size"), pct=("has_sev", "mean"))
print(bl.assign(pct=(bl["pct"] * 100).round(1)).sort_values("n", ascending=False).to_string())

print("\n" + "-" * 78)
print("PER-LABEL: of reports that MENTION the structure, how many carry a severity")
print("cue in the SAME clause? This is what the graded-target idea actually needs.")
print("-" * 78)
rows = []
clause_cache = [SPLIT.split(t) for t in tr["N"]]
for lab, cue in CUE.items():
    n_ment = n_sev = n_grade = 0
    for clauses in clause_cache:
        hit = [c for c in clauses if cue.search(c)]
        if not hit:
            continue
        n_ment += 1
        if any(ANY_SEV.search(c) for c in hit):
            n_sev += 1
        if any(GRADE.search(c) for c in hit):
            n_grade += 1
    rows.append({"label": lab, "reports_mentioning": n_ment,
                 "pct_of_corpus": n_ment / len(tr),
                 "with_severity_in_clause": n_sev / n_ment if n_ment else np.nan,
                 "with_explicit_grade": n_grade / n_ment if n_ment else np.nan})
res = pd.DataFrame(rows)
res["pct_of_corpus"] = (res["pct_of_corpus"] * 100).round(1)
res["with_severity_in_clause"] = (res["with_severity_in_clause"] * 100).round(1)
res["with_explicit_grade"] = (res["with_explicit_grade"] * 100).round(1)
print("\n" + res.to_string(index=False))

print("\n" + "-" * 78)
print("Same, but allowing the severity cue in the neighbouring clause (+/-1)")
print("-" * 78)
rows2 = []
for lab, cue in CUE.items():
    n_ment = n_sev = 0
    for clauses in clause_cache:
        idx = [i for i, c in enumerate(clauses) if cue.search(c)]
        if not idx:
            continue
        n_ment += 1
        win = set()
        for i in idx:
            win.update([i - 1, i, i + 1])
        if any(0 <= j < len(clauses) and ANY_SEV.search(clauses[j]) for j in win):
            n_sev += 1
    rows2.append({"label": lab, "mentioning": n_ment,
                  "with_severity_nearby": round(n_sev / n_ment * 100, 1) if n_ment else np.nan})
print("\n" + pd.DataFrame(rows2).to_string(index=False))

print("\n" + "=" * 78)
print("VERDICT INPUTS")
print("=" * 78)
oa = res[res.label.isin(["Medial OA", "Lateral OA", "PF OA"])]["with_severity_in_clause"].mean()
fl = res[res.label.isin(["Effusion", "Baker's"])]["with_severity_in_clause"].mean()
lg = res[res.label.isin(["ACL", "MCL"])]["with_severity_in_clause"].mean()
mn = res[res.label.str.contains("Meniscus")]["with_severity_in_clause"].mean()
print(f"\nMean 'severity in same clause' by label family:")
print(f"  OA (3 compartments)  {oa:.1f}%   <- rubric needs >50% thickness over >=1cm")
print(f"  Fluid (Effusion/Baker) {fl:.1f}%   <- rubric needs moderate-or-large")
print(f"  Ligaments (ACL/MCL)  {lg:.1f}%   <- rubric needs high-grade/complete")
print(f"  Menisci              {mn:.1f}%   <- rubric needs surfacing tear")
print("\nRead: high percentages support graded targets. Low percentages mean the")
print("reports simply do not carry the information, and the thesis weakens.")
