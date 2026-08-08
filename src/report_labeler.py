"""
Multilingual knee-MRI report labeler.

Two extractors over the same clause analysis:

  presence  - "is the finding mentioned and not negated?"   (the control)
  severity  - "would two MSK radiologists call it positive?" (the thesis)

The severity extractor exists because the official rubric is threshold-based, and the
thresholds are of two different kinds (docs/FINDINGS.md section 8):

  magnitude rubrics - OA, Effusion, Baker's, Synovitis. "moderate or large",
                      ">50% of cartilage thickness over >=1 cm". Reports answer these
                      with magnitude words 28-57% of the time.
  categorical rubrics - ACL, MCL, menisci, Contusion, Fracture. A meniscal tear is
                      graded by whether signal REACHES THE SURFACE, an ACL tear by
                      complete vs partial vs degeneration. Kind, not degree.

So group 1 is scored on a magnitude scale and group 2 on category vocabulary.

Output is a soft target in [0,1] plus a confidence weight. Soft, because the metric is
rank-based AUC: a model that learns the severity continuum ranks correctly under any
threshold, which is exactly the report-vs-image mismatch we are trying to survive.
"""
from __future__ import annotations

import re
import unicodedata

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

MAGNITUDE_LABELS = {"Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's"}
CATEGORICAL_LABELS = {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion", "Fracture"}

# Turkish dotless i and friends must fold before casefolding or "IZLENMEZ" and
# "izlenmez" diverge into different tokens.
_PRE = str.maketrans({"ı": "i", "İ": "i", "I": "i", "ß": "ss", "đ": "d", "Đ": "d",
                      "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae"})


def normalize(text: str) -> str:
    """Fold case, diacritics and separators. Greek and Cyrillic letters survive.

    NFKD strips Latin accents and Greek tonos alike (ά -> α), which is wanted: reports
    are wildly inconsistent about accents. It also folds MICRO SIGN U+00B5 to a real mu,
    which matters because many Greek reports here use the wrong codepoint for it.
    """
    if not isinstance(text, str):
        return ""
    t = text.translate(_PRE).lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.replace("­", "")
    t = re.sub(r"[_/\\]+", " ", t)
    return re.sub(r"[ \t]+", " ", t)


_SENT = re.compile(r"(?<=[.;!?])\s+|\n+")
_SUBCLAUSE = re.compile(r",|\bbut\b|\bhowever\b|\bwhereas\b|\bpero\b|\bmais\b|\bmaar\b|"
                        r"\baber\b|\bjedoch\b|\bancak\b|\bfakat\b|\bno\b ise|\bομως\b|"
                        r"\bαλλα\b|\bно\b|\bали\b|\bali\b")


def clauses(text: str) -> list[str]:
    """Split into clauses, attaching `heading:` lines to the value beneath them.

    A report line reading `Fractures :` followed by `Aucune.` is one statement. Split on
    punctuation alone and the anatomy loses its negation, flipping the label positive.
    The merged clause is a superset of the heading, so the bare heading is dropped -
    otherwise it asserts the anatomy with no negation in scope.
    """
    raw = [c.strip() for c in _SENT.split(normalize(text)) if c and c.strip()]
    out: list[str] = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c.endswith(":") and len(c.split()) <= 14 and i + 1 < len(raw):
            out.append(c + " " + raw[i + 1])
            i += 1  # consume the value; it is re-added below in its own right
            out.append(raw[i])
        else:
            out.append(c)
        i += 1
    # Long clauses hide independent assertions behind commas and contrast conjunctions.
    final = []
    for c in out:
        final.append(c)
        if len(c.split()) > 18:
            final.extend(p.strip() for p in _SUBCLAUSE.split(c) if len(p.split()) > 2)
    return final


def rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- polarity
NEG = rx(
    r"\bno\b", r"\bnot\b", r"\bwithout\b", r"\bnegative for\b", r"\babsence\b", r"\bnone\b",
    r"\bno evidence\b", r"\bunremarkable\b", r"\bfree of\b", r"\bnil\b", r"\babsent\b",
    r"\bsin\b", r"\bno hay\b", r"\bausencia\b", r"\bausentes?\b", r"\bno se observ",
    r"\bpas de\b", r"\bsans\b", r"\baucune?\b", r"\bnon\b",
    r"\bgeen\b", r"\bzonder\b", r"\bniet\b",
    r"\bkeine?n?\b", r"\bohne\b", r"\bnicht\b", r"\bkein\b",
    r"\byok\b", r"\byoktur\b", r"izlenmemekte", r"saptanmadi", r"\bdegil\b", r"\bizlenmedi\b",
    r"gozlenmemekte", r"mevcut degil", r"gorulmedi", r"\bsaptanmamistir\b",
    r"\bnema\b", r"\bbez\b", r"\bnisu\b", r"\bnije\b", r"\bne\b",
    r"\bδεν\b", r"\bχωρις\b", r"ουδεν", r"\bαρνητικ",
    r"\bбез\b", r"липсва", r"\bняма\b", r"\bне\s", r"\bотсъств",
)

NORMAL = rx(
    r"\bnormal", r"\bintact\b", r"\bpreserved\b", r"within normal limits", r"\bwnl\b",
    r"limites normales", r"\bconservad", r"\bintegr", r"\bnormales\b", r"\bhabitual",
    r"\bdoga(l|ll)", r"korunmus", r"\bnormaldir\b", r"olagan", r"\bsalim\b",
    r"\buredn", r"\bocuvan", r"\bodrzan", r"\bintakt",
    r"φυσιολογικ", r"ακεραι", r"ανευ ευρηματων",
    r"unauffallig", r"regelrecht", r"\bo\.? ?b\.?\b",
    r"нормал", r"запазен", r"съхранен", r"без особености", r"\bб\.?о\.?\b",
    r"\bgaaf\b", r"\bnormaal\b",
)

HEDGE = rx(
    r"\bpossible\b", r"\bprobable\b", r"\bsuspicious\b", r"\bsuspected\b", r"\bmay\b",
    r"cannot (be )?exclude", r"\bquestionable\b", r"\bequivocal\b", r"\blikely\b",
    r"\bposible\b", r"\bdudos", r"sin criterios categoricos", r"\bsugestiv",
    r"\bmuhtemel\b", r"\bolasi\b", r"\bsupheli\b", r"\bizlenim",
    r"\bmoguce\b", r"\bvjerojatno\b", r"\bsumnja\b",
    r"πιθαν", r"υποπτ", r"\bmoglich", r"\bverdacht", r"\bfraglich", r"\bv\.?a\.?\b",
    r"\bвъзможно\b", r"\bвероятно\b", r"суспект", r"\bmogelijk\b",
)

# --------------------------------------------------------------------------- magnitude
MAG = [
    (0.05, rx(r"\btrace\b", r"\bminimal", r"\bminim", r"\bminiem", r"\bspur\b", r"\bpunctate",
              r"ελαχιστ", r"αμελητε", r"минимал", r"\baz miktarda\b", r"\bcok az\b",
              r"\beser\b", r"\bmalko\b")),
    (0.15, rx(r"\bmild", r"\bslight", r"\bsmall\b", r"\blow.grade", r"\bleve\b", r"\bligera",
              r"\bpequen", r"\bleger", r"\bfaible\b", r"\bpetit", r"\blicht", r"\bgering",
              r"\bklein", r"\bhafif", r"\bkucuk\b", r"ηπι", r"\bμικρ", r"\bлек", r"\bмалк",
              r"\bblag", r"\bmali\b", r"\bmanji\b", r"\bdiskret", r"\bdiscret",
              r"\bsuperficial", r"\byuzeyel", r"\boppervlakkig", r"\boberflachlich")),
    (0.62, rx(r"\bmoderate", r"\bmoderad", r"\bmodere", r"\bmatig", r"\bmassig", r"\bmasig",
              r"\bmittelgradig", r"\borta\b", r"μετρι", r"\bумерен", r"\bсреден",
              r"\bumjeren", r"\bsrednj", r"\bpartial thickness")),
    (0.90, rx(r"\bsevere", r"\bmarked", r"\blarge\b", r"\bmassive", r"\badvanced",
              r"\bextensive", r"\bhigh.grade", r"\bfull.thickness", r"\bcomplete\b",
              r"\bgross\b", r"\bsevero", r"\bgrave", r"\bimportante", r"\bgrande",
              r"\bextens", r"\bavanzad", r"\bernstig", r"\bgroot", r"\buitgebreid",
              r"\bausgepragt", r"\bschwer", r"\bstark", r"\bhochgradig",
              r"\bileri\b", r"\bbelirgin", r"\bbuyuk\b", r"\byaygin", r"\bciddi",
              r"σοβαρ", r"μεγαλ", r"εκτεταμεν", r"\bтежк", r"\bизразен", r"\bголям",
              r"\btezak", r"\bizrazit", r"\bvelik", r"\bopsezn")),
]

_GRADE_NUM = re.compile(
    r"(?:grade?|grado|graad|grad|derece|evre|stadi\w*|βαθμ\w*|степен\w*|gr\.?)\s*"
    r"([1-4]|i{1,3}v?|iv)\b")
_GRADE_POST = re.compile(r"\b([1-4])\s*(?:степен|derece|\.?\s*grad)")
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
# Outerbridge/ICRS are 1-4 chondral scales; the rubric's ">50% thickness" is grade 3-4.
_GRADE_TO_SCORE = {1: 0.05, 2: 0.15, 3: 0.62, 4: 0.90}


def grade_score(clause: str):
    """Return a magnitude score from an explicit grade, or None."""
    for m in (_GRADE_NUM.search(clause), _GRADE_POST.search(clause)):
        if m:
            g = m.group(1)
            n = _ROMAN.get(g, None) if not g.isdigit() else int(g)
            if n in _GRADE_TO_SCORE:
                return _GRADE_TO_SCORE[n]
    return None


def magnitude_score(clause: str):
    """Highest magnitude cue in the clause, preferring an explicit grade. None if silent."""
    g = grade_score(clause)
    best = None
    for score, pat in MAG:
        if pat.search(clause):
            best = score if best is None else max(best, score)
    if g is not None and best is not None:
        return max(g, best)
    return g if g is not None else best


# --------------------------------------------------------------------------- anatomy
ANAT = {
    "ACL": rx(r"anterior cruciate", r"\bacl\b", r"cruzado anterior", r"\blca\b",
              r"croise anterieur", r"voorste kruisband", r"\bvkb\b",
              r"vordere[sn]? kreuzband", r"vorderen kreuzband", r"\bvkb\b",
              r"on capraz", r"\bocb\b", r"prednj\w* krizn", r"προσθι\w* χιαστ",
              r"предна кръстна", r"предната кръстна", r"передн\w* крестообраз"),
    "MCL": rx(r"medial collateral", r"\bmcl\b", r"tibial collateral",
              r"colateral (medial|interno)", r"\blcm\b", r"collateral (medial|interne)",
              r"mediale collaterale", r"binnenband", r"innenband", r"mediales? kollateral",
              r"\bic yan bag", r"medial kollateral", r"\biyb\b",
              r"medijaln\w* kolateraln", r"εσω πλαγι", r"εσωτερικο πλαγι",
              r"медиален колатерал", r"вътрешна странична"),
    "Medial Meniscus": rx(r"medial menisc", r"menisco (medial|interno)",
                          r"menisque (medial|interne)", r"mediale meniscus",
                          r"binnenmeniscus", r"innenmeniskus", r"mediale[rn]? meniskus",
                          # NOT r"\bim\b": "im" is the German preposition "in dem" and
                          # fired on nearly every German and Dutch report.
                          r"innenmeniskus\w*", r"medyal menisk", r"\bic menisk",
                          r"medijaln\w* meniskus", r"εσω μηνισκ",
                          r"медиалн\w* менискус", r"вътрешния менискус"),
    "Lateral Meniscus": rx(r"lateral menisc", r"menisco (lateral|externo)",
                           r"menisque (lateral|externe)", r"laterale meniscus",
                           r"buitenmeniscus", r"aussenmeniskus", r"laterale[rn]? meniskus",
                           r"lateral menisk", r"\bdis menisk", r"lateraln\w* meniskus",
                           r"εξω μηνισκ", r"латералн\w* менискус", r"външния менискус"),
    "Medial OA": rx(r"medial (femorotibial|tibiofemoral|compartment)",
                    r"compartimento femorotibial (medial|interno)", r"femorotibial interno",
                    r"medial (femoral|tibial) (condyle|plateau)", r"condilo femoral medial",
                    r"medialen kompartiment", r"innere[sn]? kompartiment",
                    r"medialen? (femurkondyl|tibiaplateau)", r"mediaal femorotibiaal",
                    r"medyal (femorotibial|kompartman)", r"\bic kompartman",
                    r"medijaln\w* (femorotib|kompartm|odjelj)", r"εσω διαμερισμα",
                    r"εσω (κνημιαι|μηριαι)", r"медиалн\w* (компартм|отдел|феморотиб)"),
    "Lateral OA": rx(r"lateral (femorotibial|tibiofemoral|compartment)",
                     r"compartimento femorotibial (lateral|externo)", r"femorotibial externo",
                     r"lateral (femoral|tibial) (condyle|plateau)", r"condilo femoral lateral",
                     r"lateralen kompartiment", r"aussere[sn]? kompartiment",
                     r"lateralen? (femurkondyl|tibiaplateau)", r"lateraal femorotibiaal",
                     r"lateral kompartman", r"\bdis kompartman",
                     r"lateraln\w* (femorotib|kompartm|odjelj)", r"εξω διαμερισμα",
                     r"εξω (κνημιαι|μηριαι)", r"латералн\w* (компартм|отдел|феморотиб)"),
    "PF OA": rx(r"patellofemoral", r"femoropatell", r"patelofemoral", r"retropatellar",
                r"retrorotulian", r"\btrochlea", r"\btroclea", r"\btroklea", r"\bpatell",
                r"\brotulian", r"\brotula\b", r"patellofemoraal", r"femoropatellair",
                r"επιγονατιδ", r"μηροεπιγονατιδ", r"τροχιλ", r"пател", r"феморопател"),
    "Effusion": rx(r"\beffusion", r"joint fluid", r"intra.?articular fluid", r"\bhydrops\b",
                   r"derrame articular", r"\bderrame\b", r"liquido articular",
                   r"epanchement", r"gewrichtsvocht", r"gewrichtseffusie",
                   r"gelenkergu", r"\bergu(ss|s)\b", r"eklem\w* ic\w* sivi", r"efuzyon",
                   r"eklem sivisi", r"sivi (miktari|artisi|birikimi)", r"\bsivi artis",
                   r"\bizljev", r"\bizliv", r"zglobn\w* tekucin", r"\bhidrops\b",
                   r"αρθρικ\w* υγρ", r"ενδαρθρικ", r"αρθρικη συλλογη", r"ποσοτητα υγρου",
                   r"ставен излив", r"\bизлив", r"ставна течност", r"синовиална течност"),
    "Synovitis": rx(r"synovit", r"sinovit", r"synovial (thickening|proliferation|hypertroph)",
                    r"synoviale? (verdikking|proliferatie)", r"synovialitis",
                    r"synovialis(verdickung|proliferation)", r"sinovijalitis",
                    r"zadebljanje sinovij", r"υμενιτιδ", r"συνοβιτιδ", r"υμενικ\w* υπερτροφ",
                    r"синовит", r"синовиал\w* (задебел|пролифер)", r"\bpannus\b"),
    "Baker's": rx(r"baker", r"popliteal cyst", r"quiste popliteo", r"quistes popliteos",
                  r"kyste poplite", r"popliteale? cyst", r"poplitealzyste", r"bakerzyste",
                  r"popliteal kist", r"bakerova", r"poplitealn\w* cist",
                  r"κυστη baker", r"κυστη του baker", r"бейкер", r"поплитеал\w* киста"),
    "Contusion": rx(r"contusion", r"bone bruise", r"bone marrow (o?edema|edema)",
                    r"marrow (o?edema)", r"edema oseo", r"contusion osea", r"oedeme osseux",
                    r"botoedeem", r"beenmergoedeem", r"knochenmarksodem", r"knochenodem",
                    r"kemik ilig\w* odem", r"kontuzyon", r"kemik odem",
                    r"kostn\w* edem", r"οστικο οιδημα", r"οιδημα μυελου",
                    r"костномозъчен едем", r"костен едем", r"оток на костния"),
    "Fracture": rx(r"fractur", r"fraktur", r"fractura", r"\bbreuk\b", r"\bkirik", r"prijelom",
                   r"καταγμα", r"κατεαγ", r"фрактур", r"счупв"),
}

# Evidence that a compartment actually has osteoarthritis, as opposed to merely existing.
OA_EVID = rx(
    r"osteoarthrit", r"\barthros", r"gonarthros", r"osteoarthros", r"chondropath",
    r"chondromalac", r"condropat", r"condromalac", r"cartilage (loss|thinning|defect)",
    r"chondral (loss|defect|ulcer|thinning|fissur)", r"osteophyt", r"osteofit", r"osteofyt",
    r"joint space narrowing", r"pinzamiento articular", r"kikirdak (kayb|incelme)",
    r"kondropati", r"kondral", r"kraakbeen(lijden|verlies)", r"gonartrose", r"artrose",
    r"knorpel(verlust|schaden|defekt|lasion)", r"arthrose", r"gonarthrose", r"chondropathie",
    r"hrskavic", r"hondromalac", r"artroz", r"osteoartrit", r"stanjenje",
    r"χονδρ\w*παθ", r"αρθριτ", r"αρθρωσ", r"οστεοφυτ", r"αρθρικου χονδρου",
    r"артроз", r"хондропат", r"остеофит", r"хрущял", r"изтън",
    r"ulcera\w* condral", r"outerbridge", r"\bicrs\b", r"kellgren", r"\bkraakbeen\b",
)

# --------------------------------------------------------------- categorical vocabulary
TEAR = rx(r"\btear", r"\btorn\b", r"\brupture", r"\bdisruption\b", r"discontinuit",
          r"\bavuls", r"\brotura", r"\bruptura", r"\bdesgarro", r"\broto\b",
          r"\bdechirure", r"\bdechire", r"\bscheur", r"\bruptuur", r"gescheurd",
          r"riss", r"einriss", r"\bruptur", r"zerreiss", r"\byirtik", r"\byirtig",
          r"\bkopma\b", r"butunluk kaybi", r"\bpuknuce", r"\bprekid\b", r"\bpukotin",
          r"ρηξη", r"ρηξις", r"ρηγμα", r"руптура", r"разкъсв", r"разрив", r"скъсв")

# Meniscus: the rubric is "signal DEFINITELY contacts the surface on >=2 images".
SURFACING = rx(r"reach\w* the (articular )?surface", r"contact\w* the surface", r"surfacing",
               r"extend\w* to the (articular |inferior |superior )?surface",
               r"communicat\w* with the (articular )?surface", r"\bgrade 3\b", r"\bgrade iii\b",
               r"alcanza la superficie", r"contacta (con )?la superficie",
               r"atteint la surface", r"bereikt het oppervlak",
               r"oberflachenkontakt", r"erreicht die oberflache",
               r"yuzeye ulas", r"eklem yuzune", r"dodiruje (zglobnu )?povrsinu",
               r"φθανει (στην|την) (αρθρικη )?επιφανεια", r"достига (до )?повърхността")
INTRASUBSTANCE = rx(r"intrasubstance", r"does not reach the surface",
                    r"without surfacing", r"no surfacing", r"\bgrade 2\b", r"\bgrade ii\b",
                    r"degenerative signal", r"mucoid", r"myxoid", r"\bfray",
                    r"intrasustancia", r"no alcanza la superficie", r"degenerativ\w* senal",
                    r"intrasubstantiel", r"n.atteint pas la surface",
                    r"intrasubstantieel", r"bereikt niet", r"binnenin",
                    r"intrasubstanziell", r"erreicht (die oberflache )?nicht",
                    r"dejeneratif", r"yuzeye ulasmayan", r"intrasupstanc",
                    r"εκφυλ", r"δεν φθανει", r"дегенерат", r"не достига",
                    r"\b2\s*степен", r"\bстепен\s*2", r"\b2a\b", r"\b2b\b")
DISPLACED = rx(r"bucket.handle", r"displaced fragment", r"\bflap tear", r"root tear",
               r"radial tear", r"\bparrot beak", r"asa de cubo", r"anse de seau",
               r"emmerhandvat", r"korbhenkel", r"kova sapi", r"δικην λαβης",
               r"кофа", r"разместен")

# ACL / MCL: complete vs partial vs degeneration, and acute vs chronic for MCL.
COMPLETE = rx(r"complete (tear|rupture|disruption)", r"full.thickness tear", r"high.grade tear",
              r"total (rupture|tear)", r"rotura completa", r"ruptura completa",
              r"rupture complete", r"complete ruptuur", r"komplettruptur", r"totalruptur",
              r"tam kopma", r"komplet(na)? ruptur", r"πληρης ρηξη", r"пълна руптура",
              r"\bdiscontinuity\b", r"fibers? (are )?disrupted", r"non.?visualiz")
PARTIAL = rx(r"partial (tear|rupture|thickness)", r"\bpartial\b", r"low.grade tear",
             r"rotura parcial", r"parcial", r"rupture partielle", r"partiele ruptuur",
             r"partialruptur", r"teilruptur", r"parsiyel", r"kismi",
             r"parcijaln", r"μερικη ρηξη", r"частичн")
CHRONIC = rx(r"\bchronic", r"\bold\b", r"\bremote\b", r"\bhealed\b", r"\bprior\b",
             r"post.?operative", r"reconstruct", r"\bcronic", r"antigua", r"\bancien",
             r"\bchronisch", r"\bkronik", r"\beski\b", r"\bkronicn", r"χρονι", r"стар",
             r"\bsequela", r"\bscar", r"fibros")
SPRAIN = rx(r"\bsprain", r"grade 1", r"grade i\b", r"\besguince", r"\bentorse", r"\bentorsis",
            r"\bzerrung", r"\bdistorsion", r"\bburkulma", r"\bistegnuce", r"διαστρεμμα")

# Fracture: the rubric wants an ACUTE CORTICAL BREAK. Osteochondral and stress
# injuries are host-flagged as not counting.
FRAC_ACUTE = rx(r"acute fractur", r"cortical (break|disruption)", r"fracture line",
                r"\bfractura aguda", r"linea de fractura", r"trait de fracture",
                r"frakturlinie", r"akut\w* fraktur", r"akut kirik", r"fraktur linij",
                r"γραμμη καταγματος", r"οξυ καταγμα", r"фрактурна линия")
FRAC_EXCLUDE = rx(r"osteochondral fractur", r"stress fractur", r"insufficiency fractur",
                  r"subchondral fractur", r"avulsion fractur", r"\bosteocondral",
                  r"fractura de estres", r"stressfraktur", r"stres kirig")

# Contusion: marrow oedema from impact, WITHOUT a discrete fracture line.
IMPACT = rx(r"contusion", r"bruise", r"\bimpact", r"pivot.shift", r"kissing",
            r"traumatic", r"\btraumat", r"\bcontusion osea", r"\btravmat", r"\bkontuzyon")

# Synovitis is scored on images in ~47% of the gold studies but named in only ~12% of
# reports (docs/FINDINGS.md section 9). Clinical radiologists rarely write the word, but
# they do describe its imaging signs. These proxies are weaker evidence than the term
# itself and are scored lower, with lower confidence.
SYNOV_PROXY = rx(
    r"hoffa", r"infrapatellar fat pad (o?edema|inflam)", r"fat pad (o?edema|inflam)",
    r"\bbursitis", r"suprapatellar burs\w* (distend|fluid|thicken)", r"bursite",
    r"\bbursitis\b", r"schleimbeutelentzundung", r"bursit", r"θυλακιτιδα", r"бурсит",
    r"capsul\w* (thicken|hypertroph)", r"kapsel(verdickung|hypertroph)",
    r"engrosamiento capsular", r"kapsul\w* kalinlas", r"θυλακ\w* παχυν",
    r"villonodular", r"\bpvns\b", r"vellonodular",
    r"synovi\w* enhanc", r"realce sinovial", r"synovial (fluid )?debris",
    r"complex (joint )?fluid", r"loculated", r"septat",
    r"\bplica\b(?=[^.]{0,40}(inflam|thicken|edema))",
)

# Where an unmentioned finding sits in the within-label ordering.
#
# An earlier version set these to P(positive | not mentioned) - 0.18 for Synovitis, to
# express "the report under-reports this badly". That cost 0.121 AUC on Synovitis, and
# the reason is instructive: AUC is computed PER LABEL, so the absolute value carries no
# information at all. Only the rank matters. Setting the prior at 0.18 put every silent
# study ABOVE a report that said "mild synovitis" at 0.15 - an inversion, since a mention
# is evidence FOR the finding.
#
# So severity carries the best rank estimate, and `confidence` - not an inflated
# severity - carries the uncertainty. Silence sits just above explicit negation (0.02)
# and below any genuine mention (>=0.10).
UNMENTIONED_PRIOR = {
    "Synovitis": 0.06,      # 46.6% gold, named in 11.9% of reports - the extreme case
    "Fracture": 0.05, "Medial OA": 0.05, "Lateral OA": 0.05, "PF OA": 0.05,
    "Contusion": 0.05, "Baker's": 0.04, "Effusion": 0.04,
    "ACL": 0.04, "MCL": 0.04, "Medial Meniscus": 0.04, "Lateral Meniscus": 0.04,
}
# A short report omits negatives, so silence in one is weaker evidence of absence than
# silence in a long structured report. That IS rank information, so it earns a bump -
# but the bump must stay below a genuine mention.
SHORT_REPORT_PRIOR = 0.09


def _polarity(clause: str, finding_asserted: bool = False):
    """(negated, hedged) for a clause.

    `NORMAL` ("intact", "normal", "unauffallig") asserts normality, but a clause can
    assert a finding AND close with a normality remark - "moderate effusion, remainder
    normal". Treating that as negated zeroes a real positive, so normality only negates
    when the caller has found no finding of its own. An explicit NEG always negates.
    """
    hedge = bool(HEDGE.search(clause))
    if NEG.search(clause):
        return True, hedge
    if NORMAL.search(clause) and not finding_asserted:
        return True, hedge
    return False, hedge


def extract_side(text: str):
    """Which knee, from the report. Training-only - used to calibrate the geometry rule.

    docs/FINDINGS.md section 1: half of all studies carry no DICOM Laterality tag, and
    GE MEDICAL SYSTEMS carries none at all, so the geometry sign rule cannot be validated
    on GE without a third source. Reports name the side in their first line.
    """
    t = normalize(text)[:400]
    left = rx(r"\bleft\b", r"\bizquierd", r"\bgauche\b", r"\blinks\b", r"\blinke[srn]?\b",
              r"\bsol\b", r"\blijev", r"\blijeva", r"\bαριστερ", r"\bляв", r"\blinker\b",
              r"\bl\.? diz\b", r"\bl knee\b")
    right = rx(r"\bright\b", r"\bderech", r"\bdroit", r"\brechts\b", r"\brechte[srn]?\b",
               r"\bsag\b", r"\bdesn", r"\bδεξι", r"\bдесн", r"\br\.? diz\b", r"\br knee\b")
    hl, hr = bool(left.search(t)), bool(right.search(t))
    if hl and not hr:
        return "L"
    if hr and not hl:
        return "R"
    if hl and hr:
        return "B"
    return None


def _label_clauses(cls: list[str], label: str):
    """Clauses mentioning this label's anatomy, with OA scoped to real OA evidence."""
    pat = ANAT[label]
    hits = [c for c in cls if pat.search(c)]
    if label in ("Medial OA", "Lateral OA", "PF OA"):
        # A compartment being named is not osteoarthritis. Require evidence.
        hits = [c for c in hits if OA_EVID.search(c)]
    if label == "Synovitis":
        # Fall back to imaging signs when the word itself is absent.
        proxy = [c for c in cls if SYNOV_PROXY.search(c) and c not in hits]
        return hits, proxy
    return hits, []


def _score_magnitude(label: str, hits: list[str]) -> tuple[float, float]:
    """Group 1: the rubric asks 'how much?'. Returns (score, confidence)."""
    best, conf = 0.0, 0.5
    for c in hits:
        m = magnitude_score(c)
        neg, hedge = _polarity(c, finding_asserted=m is not None)
        if neg:
            best = max(best, 0.02)
            conf = max(conf, 0.9)
            continue
        if m is None:
            # Asserted but ungraded. The rubric needs moderate-or-large, and "on the
            # fence" was graded negative, so an ungraded mention sits below the middle.
            s, cf = 0.38, 0.45
        else:
            s, cf = m, 0.85
        if hedge:
            s *= 0.55
            cf *= 0.7
        if s > best:
            best, conf = s, cf
    return best, conf


def _score_categorical(label: str, hits: list[str]) -> tuple[float, float]:
    """Group 2: the rubric asks 'what kind?'. Returns (score, confidence).

    Negation is tested FIRST, exactly as in `_score_magnitude`. An earlier version put
    it last, so "medial meniscus: no tear" matched TEAR and scored 0.72 - the negation
    branch was unreachable whenever any pathology word appeared, which is nearly always.
    That single ordering mistake is why the magnitude family gained +0.049 on the 58 gold
    studies while this family gained +0.000.
    """
    best, conf = 0.0, 0.5
    for c in hits:
        asserted = bool(TEAR.search(c) or COMPLETE.search(c) or SURFACING.search(c)
                        or DISPLACED.search(c) or FRAC_ACUTE.search(c) or IMPACT.search(c))
        neg, hedge = _polarity(c, finding_asserted=asserted)
        if neg:
            best = max(best, 0.02)
            conf = max(conf, 0.9)
            continue

        s = cf = None
        if label in ("Medial Meniscus", "Lateral Meniscus"):
            if DISPLACED.search(c):
                s, cf = 0.95, 0.9
            elif SURFACING.search(c):
                s, cf = 0.88, 0.85
            elif INTRASUBSTANCE.search(c) and not TEAR.search(c):
                # Grade 2 / intrasubstance signal is explicitly NEGATIVE in the rubric -
                # but only when no tear is asserted alongside it. "Complex degenerative
                # tear" is a tear that happens to be degenerative, not a non-tear.
                s, cf = 0.10, 0.8
            elif TEAR.search(c):
                s, cf = 0.72, 0.6
            elif INTRASUBSTANCE.search(c):
                s, cf = 0.10, 0.7

        elif label in ("ACL", "MCL"):
            if COMPLETE.search(c):
                s, cf = 0.93, 0.9
            elif PARTIAL.search(c) and TEAR.search(c):
                s, cf = 0.45, 0.7
            elif SPRAIN.search(c):
                s, cf = 0.08, 0.75     # low-grade sprain is negative in the rubric
            elif TEAR.search(c):
                s, cf = 0.70, 0.6
            if s is not None and label == "MCL" and CHRONIC.search(c):
                s *= 0.35              # rubric wants an ACUTE MCL tear
                cf = max(cf, 0.7)

        elif label == "Fracture":
            if FRAC_EXCLUDE.search(c):
                s, cf = 0.22, 0.7      # osteochondral/stress: host-flagged as not counting
            elif FRAC_ACUTE.search(c):
                s, cf = 0.92, 0.9
            elif CHRONIC.search(c):
                s, cf = 0.08, 0.7
            elif ANAT["Fracture"].search(c):
                s, cf = 0.65, 0.55

        elif label == "Contusion":
            if IMPACT.search(c):
                s, cf = 0.85, 0.85
            else:
                # marrow oedema with no impact word: may be degenerative rather than
                # traumatic, and the rubric wants impact.
                m = magnitude_score(c)
                s, cf = (0.45 if m is None else max(0.30, m)), 0.5

        if s is None:
            continue
        if hedge:
            s *= 0.55
            cf *= 0.7
        if s > best:
            best, conf = s, cf
    return best, conf


def label_report(text: str, n_words: int | None = None) -> dict:
    """Score all 12 labels. Returns {label: {presence, severity, confidence}}.

    `n_words` drives the unmentioned prior. docs/FINDINGS.md section 8: 34% of
    Croatian/Flemish and 29% of Spanish reports run under 50 words and simply do not
    list negatives, so "not mentioned" means something different per site - and site
    tracks language. Mapping it to a hard 0 would inject site-correlated label noise.
    """
    cls = clauses(text)
    if n_words is None:
        n_words = len(normalize(text).split())
    short = n_words < 60
    out = {}
    for label in LABELS:
        hits, proxy = _label_clauses(cls, label)

        if not hits and not proxy:
            # Silence in a short report is weaker evidence of absence than silence in a
            # long structured one. Uncertainty lives in `confidence`, not in `severity`.
            prior = UNMENTIONED_PRIOR.get(label, 0.04)
            out[label] = {"presence": 0,
                          "severity": SHORT_REPORT_PRIOR if short else prior,
                          "confidence": 0.2 if short else 0.6,
                          "mentioned": False}
            continue

        if hits:
            if label in MAGNITUDE_LABELS:
                sev, conf = _score_magnitude(label, hits)
            else:
                sev, conf = _score_categorical(label, hits)
        else:
            sev, conf = 0.0, 0.4

        if proxy:
            # Imaging signs of synovitis, scored below an explicit mention.
            psev, _ = _score_magnitude(label, proxy)
            psev = min(psev, 0.55)
            if psev > sev:
                sev, conf = psev, 0.45

        pres = 0
        for c in hits:
            asserted = bool(TEAR.search(c) or COMPLETE.search(c) or SURFACING.search(c))
            neg, _ = _polarity(c, finding_asserted=asserted)
            if neg:
                continue
            if label in MAGNITUDE_LABELS or TEAR.search(c) or ANAT[label].search(c):
                pres = 1
                break
        out[label] = {"presence": pres, "severity": round(float(sev), 4),
                      "confidence": round(float(conf), 3),
                      "mentioned": bool(hits)}
    return out
