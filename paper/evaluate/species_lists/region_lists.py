# Written with Claude (Anthropic) under the author's direction; see paper/README.md.
"""regionlists_v1: two foreign species lists (Germany, Costa Rica) from published national checklists.

The lists are fetched keyless from Wikipedia at a pinned revision (both pages transcribe the national
committee lists: DO-G 'Liste der Vögel Deutschlands', IOC 9.2 names; AOCR 'Lista oficial de las aves
de Costa Rica', AOS names). Rules (regionlists_v1):
  DE  categories A and C (natural records since 1950 + established introductions); B-only, D, E dropped
  CR  every species except extirpated ones; accidentals kept
Scientific name -> eBird 2024 code (exact) -> Perch class (teacher.embed_perch.perch_classes); a name without an
eBird 2024 match (an older synonym) takes the unique eBird 2024 species with the same epithet stem
(first five letters, so gender endings agree) in a family that holds the old genus, or an explicit
entry of SYNONYMS; names matching
neither stay dead rows (vocab_index -1), as in kinglet build-pack. Every synonym match is printed.

  python evaluate/species_lists/region_lists.py      # -> results/regionlist_{de,cr}.csv (ebird_code, scientific_name, vocab_index)
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.config import RESULTS  # noqa: E402
from common.scoring import EBIRD_TAXONOMY  # noqa: E402
from teacher.embed_perch import perch_classes  # noqa: E402

LISTS = {"de": ("List_of_birds_of_Germany", 1377455216), "cr": ("List_of_birds_of_Costa_Rica", 1356091564)}
API = "https://en.wikipedia.org/w/api.php?action=parse&oldid={rev}&prop=wikitext&format=json&formatversion=2"
# older names the stem rule cannot resolve (genus and epithet both changed, lumps, typos) -> eBird 2024 name
SYNONYMS = {
    "Anas carolinensis": "Anas crecca", "Microcarbo pygmeus": "Microcarbo pygmaeus",
    "Ixobrychus minutus": "Botaurus minutus", "Ixobrychus exilis": "Botaurus exilis", "Bubulcus ibis": "Ardea ibis",
    "Acanthis cabaret": "Acanthis flammea", "Acanthis hornemanni": "Acanthis flammea",
    "Sarkidiornis silvicola": "Sarkidiornis sylvicola", "Micropygia schomburgkii": "Rufirallus schomburgkii",
    "Hapalocrex flaviventer": "Laterallus flaviventer", "Drobates fumigatus": "Dryobates fumigatus",
    "Veniliornis kirkii": "Dryobates kirkii", "Milvago chimachima": "Daptrius chimachima",
    "Calocitta formosa": "Cyanocorax formosus", "Psilorhinus morio": "Cyanocorax morio",
    "Spinus xanthogastra": "Spinus xanthogastrus",
}
LINE = re.compile(r"^\*.*?''([A-Z][a-z]+ [a-z-]+)''(.*)$")


def keep(region: str, tail: str) -> bool:
    if region == "de":  # tail: " – A", " – BC", " – A<ref .../>"
        cat = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", tail).strip(" –-")
        return bool(set(cat) & {"A", "C"}) and set(cat) <= {"A", "B", "C"}
    return "exterpated" not in tail and "extirpated" not in tail.lower()


def parse(region: str, wikitext: str) -> list[str]:
    if region == "de":
        wikitext = wikitext.split("==Appendices==")[0]  # categories D and E
    names = []
    for ln in wikitext.splitlines():
        m = LINE.match(ln)
        if m and keep(region, m.group(2)):
            names.append(m.group(1))
    assert len(names) == len(set(names)), "duplicate species line"
    return names


def main():
    tax = pd.read_csv(EBIRD_TAXONOMY, usecols=["CATEGORY", "SPECIES_CODE", "SCI_NAME", "FAMILY"])
    tax = tax[tax.CATEGORY == "species"]
    sci2code = tax.set_index("SCI_NAME")["SPECIES_CODE"].to_dict()
    genus, epithet = tax.SCI_NAME.str.split().str[0], tax.SCI_NAME.str.split().str[1]

    assert all(v in sci2code for v in SYNONYMS.values())

    def synonym(name: str) -> str:
        g, e = name.split()
        fams = set(tax.FAMILY[genus == g])
        hit = tax[epithet.str[:5].eq(e[:5]) & (tax.FAMILY.isin(fams) | genus.eq(g))]
        return hit.SPECIES_CODE.iloc[0] if len(hit) == 1 else ""
    for region, (page, rev) in LISTS.items():
        req = urllib.request.Request(API.format(rev=rev), headers={"User-Agent": "research-script/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)["parse"]
        assert d["revid"] == rev
        names = parse(region, d["wikitext"])
        codes = [sci2code.get(n) or sci2code.get(SYNONYMS.get(n)) or synonym(n) for n in names]
        pidx = {c["species"]: c["perch_idx"] for c in perch_classes([c for c in codes if c])}
        df = pd.DataFrame({"ebird_code": [c or f"nomatch_{n.replace(' ', '_')}" for c, n in zip(codes, names)],
                           "scientific_name": names, "vocab_index": [pidx.get(c, -1) for c in codes]})
        df = df.drop_duplicates("ebird_code")
        out = RESULTS / f"regionlist_{region}.csv"
        df.to_csv(out, index=False)
        print(f"{region}: {page} rev {rev}: {len(names)} species, {sum(n in sci2code for n in names)} eBird 2024 names, {sum(map(bool, codes))} with a code, "
              f"{int((df.vocab_index >= 0).sum())} with a Perch class -> {out}")
        code2sci = tax.set_index("SPECIES_CODE")["SCI_NAME"]
        print("  synonyms:", [f"{n} -> {code2sci[c]}" for n, c in zip(names, codes) if c and n not in sci2code])
        print("  no eBird match:", [n for n, c in zip(names, codes) if not c])
        print("  no Perch class:", df.loc[(df.vocab_index < 0) & ~df.ebird_code.str.startswith("nomatch"),
                                          "ebird_code"].tolist())


if __name__ == "__main__":
    main()
