"""EDA #2: proper language ID + sample reports per language + gold-label report inspection."""
import pandas as pd, numpy as np, re, collections

D = r"H:\RSNA Knee Abnormality Detection\data"
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
tr = pd.read_csv(f"{D}/train.csv")

RANGES = [
    ("Greek",      r"[\u0370-\u03ff]"),
    ("Cyrillic",   r"[\u0400-\u04ff]"),
    ("Han",        r"[\u4e00-\u9fff]"),
    ("Kana",       r"[\u3040-\u30ff]"),
    ("Hangul",     r"[\uac00-\ud7af]"),
    ("Arabic",     r"[\u0600-\u06ff]"),
    ("Hebrew",     r"[\u0590-\u05ff]"),
    ("Thai",       r"[\u0e00-\u0e7f]"),
    ("Devanagari", r"[\u0900-\u097f]"),
]
# stopword fingerprints for Latin-script languages
STOP = {
 "English":    ["the","and","with","normal","tear","there","without","joint","is "],
 "Spanish":    ["del","con","sin","que","los","las","señal","rotura","menisco","articular"],
 "French":     ["du ","des","avec","sans","une","est","genou","ménisque","lésion","rupture"],
 "Portuguese": ["do ","da ","com","sem","não","joelho","menisco","lesão","sinal"],
 "German":     ["der","die","und","mit","ohne","nicht","kein","gelenk","kniegelenk"],
 "Italian":    ["del","con","non","una","ginocchio","menisco","lesione","segnale"],
 "Dutch":      ["van","het","een","niet","geen","met","knie"],
 "Turkish":    ["ve ","bir","ile","dizde","menisküs","yirtik","normal"],
 "Polish":     ["nie","jest","oraz","staw","kolana","wiez"],
}

def script(s):
    for name, pat in RANGES:
        if len(re.findall(pat, s)) > 5:
            return name
    return "Latin"

def latin_lang(s):
    low = " " + s.lower() + " "
    best, bs = "Unknown", 0
    for lang, words in STOP.items():
        sc = sum(low.count(" "+w.strip()+" ") if len(w.strip())>1 else 0 for w in words)
        sc += sum(low.count(w) for w in words if len(w.strip())>4)
        if sc > bs: best, bs = lang, sc
    return best if bs >= 3 else "Latin-other"

tr["script"] = tr["Report"].astype(str).map(script)
tr["lang"] = np.where(tr["script"]=="Latin",
                      tr["Report"].astype(str).map(latin_lang),
                      tr["script"])

print("="*70); print("LANGUAGE / SCRIPT DISTRIBUTION (all 4407 studies)")
vc = tr["lang"].value_counts()
print(pd.DataFrame({"n": vc, "pct": (vc/len(tr)*100).round(1)}).to_string())

print("\n" + "="*70); print("LANGUAGE OF THE 58 GOLD-LABELED STUDIES")
gold = tr[tr[LABELS].notna().all(axis=1)]
print(gold["lang"].value_counts().to_string())

print("\n" + "="*70); print("SAMPLE REPORTS (first 700 chars) — one per language")
for lang in vc.index:
    sub = tr[tr["lang"]==lang]
    r = sub["Report"].iloc[0]
    print("\n" + "-"*70)
    print(f"### {lang}  (n={len(sub)})   study={sub['StudyInstanceUID'].iloc[0][-12:]}")
    print("-"*70)
    print(str(r)[:700].replace("\r"," "))

print("\n" + "="*70)
print("FULL REPORTS FOR 3 GOLD-LABELED STUDIES (with their labels)")
for i in range(3):
    row = gold.iloc[i]
    pos = [l for l in LABELS if row[l]==1]
    neg = [l for l in LABELS if row[l]==0]
    print("\n" + "="*70)
    print(f"lang={row['lang']}  POSITIVE={pos}")
    print(f"NEGATIVE={neg}")
    print("-"*70)
    print(str(row["Report"])[:2500].replace("\r"," "))

tr[["StudyInstanceUID","lang","script"]].to_csv(r"H:\RSNA Knee Abnormality Detection\eda\report_lang.csv", index=False)
print("\n\n[saved eda/report_lang.csv]")
