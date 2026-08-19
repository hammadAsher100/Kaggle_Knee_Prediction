"""Versioned multilingual terminology for the discovered competition targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetTerminology:
    terms: tuple[str, ...]
    requires_pathology_marker: bool = False


PATHOLOGY_MARKERS: dict[str, tuple[str, ...]] = {
    "en": ("tear", "torn", "rupture", "sprain", "injury", "degeneration", "abnormal"),
    "es": ("rotura", "ruptura", "desgarro", "lesión", "esguince", "degeneración"),
    "fr": ("rupture", "lésion", "entorse", "dégénérescence", "anormal"),
    "de": ("riss", "ruptur", "läsion", "zerrung", "degeneration", "pathologisch"),
    "nl": ("scheur", "ruptuur", "letsel", "verrekking", "degeneratie"),
    "tr": ("yırtık", "yirtik", "rüptür", "ruptur", "yaralanma", "dejenerasyon"),
    "bg": ("разкъсване", "руптура", "увреда", "дегенерация"),
    "el": ("ρήξη", "κακωση", "κάκωση", "εκφύλιση"),
    "hr": ("ruptura", "lezija", "oštećenje", "ostecenje", "degeneracija"),
    "unknown": ("tear", "rupture", "ruptura", "riss", "lésion", "lezija"),
}


TARGET_TERMINOLOGY: dict[str, TargetTerminology] = {
    "ACL": TargetTerminology(
        (
            "acl",
            "anterior cruciate ligament",
            "ligamento cruzado anterior",
            "lca",
            "ligament croisé antérieur",
            "vorderes kreuzband",
            "vkb",
            "voorste kruisband",
            "ön çapraz bağ",
            "öçb",
            "предна кръстна връзка",
            "пкс",
            "πρόσθιος χιαστός",
            "πχσ",
            "prednji križni ligament",
            "prednji ukriženi ligament",
        ),
        requires_pathology_marker=True,
    ),
    "MCL": TargetTerminology(
        (
            "mcl",
            "medial collateral ligament",
            "ligamento colateral medial",
            "lcm",
            "ligament collatéral médial",
            "mediales kollateralband",
            "innenband",
            "mediale collaterale band",
            "medial kollateral bağ",
            "iç yan bağ",
            "медиална колатерална връзка",
            "έσω πλάγιος σύνδεσμος",
            "medijalni kolateralni ligament",
        ),
        requires_pathology_marker=True,
    ),
    "Medial Meniscus": TargetTerminology(
        (
            "medial meniscus",
            "menisco medial",
            "ménisque médial",
            "innenmeniskus",
            "mediale meniscus",
            "medial menisküs",
            "медиален менискус",
            "έσω μηνίσκος",
            "medijalni menisk",
        ),
        requires_pathology_marker=True,
    ),
    "Lateral Meniscus": TargetTerminology(
        (
            "lateral meniscus",
            "menisco lateral",
            "ménisque latéral",
            "außenmeniskus",
            "laterale meniscus",
            "lateral menisküs",
            "латерален менискус",
            "έξω μηνίσκος",
            "lateralni menisk",
        ),
        requires_pathology_marker=True,
    ),
    "Medial OA": TargetTerminology(
        (
            "medial compartment osteoarthritis",
            "medial compartment arthrosis",
            "medial compartment chondrosis",
            "medial gonarthrosis",
            "medial tibiofemoral osteoarthritis",
            "artrosis femorotibial medial",
            "arthrose fémoro-tibiale médiale",
            "mediale gonarthrose",
            "mediale compartimentsartrose",
            "medial kompartman osteoartrit",
            "медиална гонартроза",
            "οστεοαρθρίτιδα έσω διαμερίσματος",
            "medijalna gonartroza",
        )
    ),
    "Lateral OA": TargetTerminology(
        (
            "lateral compartment osteoarthritis",
            "lateral compartment arthrosis",
            "lateral compartment chondrosis",
            "lateral gonarthrosis",
            "lateral tibiofemoral osteoarthritis",
            "artrosis femorotibial lateral",
            "arthrose fémoro-tibiale latérale",
            "laterale gonarthrose",
            "laterale compartimentsartrose",
            "lateral kompartman osteoartrit",
            "латерална гонартроза",
            "οστεοαρθρίτιδα έξω διαμερίσματος",
            "lateralna gonartroza",
        )
    ),
    "PF OA": TargetTerminology(
        (
            "patellofemoral osteoarthritis",
            "patellofemoral arthrosis",
            "patellofemoral chondrosis",
            "retropatellar chondrosis",
            "artrosis femoropatelar",
            "arthrose fémoro-patellaire",
            "retropatellararthrose",
            "patellofemorale artrose",
            "patellofemoral osteoartrit",
            "пателофеморална артроза",
            "επιγονατιδομηριαία οστεοαρθρίτιδα",
            "patelofemoralna artroza",
        )
    ),
    "Effusion": TargetTerminology(
        (
            "joint effusion",
            "knee effusion",
            "effusion",
            "derrame articular",
            "derrame",
            "épanchement articulaire",
            "épanchement",
            "gelenkerguss",
            "erguss",
            "gewrichtseffusie",
            "eklem efüzyonu",
            "efüzyon",
            "ставен излив",
            "αρθρική συλλογή",
            "izljev u zglobu",
        )
    ),
    "Synovitis": TargetTerminology(
        (
            "synovitis",
            "sinovitis",
            "synovite",
            "synovialitis",
            "synoviitis",
            "sinovit",
            "синовит",
            "υμενίτιδα",
        )
    ),
    "Baker's": TargetTerminology(
        (
            "baker cyst",
            "baker's cyst",
            "popliteal cyst",
            "quiste de baker",
            "kyste de baker",
            "baker-zyste",
            "bakerse cyste",
            "baker kisti",
            "киста на бейкър",
            "κύστη baker",
            "bakerova cista",
        )
    ),
    "Contusion": TargetTerminology(
        (
            "bone contusion",
            "bone bruise",
            "osseous contusion",
            "contusión ósea",
            "contusion osseuse",
            "knochenkontusion",
            "botcontusie",
            "kemik kontüzyonu",
            "костна контузия",
            "οστική θλάση",
            "koštana kontuzija",
        )
    ),
    "Fracture": TargetTerminology(
        (
            "fracture",
            "fractura",
            "fraktur",
            "fractuur",
            "kırık",
            "kirik",
            "фрактура",
            "счупване",
            "κάταγμα",
            "prijelom",
        )
    ),
}


def pathology_markers(language: str) -> tuple[str, ...]:
    return PATHOLOGY_MARKERS.get(language, ()) + PATHOLOGY_MARKERS["unknown"]
