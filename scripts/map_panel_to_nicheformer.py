"""Map a mouse gene panel (symbols, one per line) to Nicheformer vocabulary tokens (human orthologue Ensembl ids) through Ensembl REST:
upper-cased symbol -> human gene; for the rest, mouse symbol -> mouse id -> one2one human orthologue.
    python scripts/map_panel_to_nicheformer.py <panel.txt> <out.json> <nicheformer_hf/vocab.json>"""
import json, sys, time, requests
panel = [l.strip() for l in open(sys.argv[1]) if l.strip()]; OUT = sys.argv[2]
vocab = json.load(open(sys.argv[3]))
S = requests.Session(); S.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
def post(url, symbols):
    for attempt in range(5):
        r = S.post(url, json={"symbols": symbols}, timeout=180)
        if r.status_code == 200:
            try: return r.json()
            except Exception: pass
        print("  retry", url.split("/")[-1], r.status_code, r.text[:80].replace("\n", " "), flush=True); time.sleep(3 + 3 * attempt)
    return {}
hs = {}
for i in range(0, len(panel), 250): hs.update(post("https://rest.ensembl.org/lookup/symbol/homo_sapiens", [g.upper() for g in panel[i:i + 250]]))
mapping = {g: {"ensembl": hs.get(g.upper(), {}).get("id"), "how": "upper_symbol"} for g in panel}
for g in panel:
    if mapping[g]["ensembl"] not in vocab: mapping[g] = {"ensembl": None, "how": None}
miss = [g for g in panel if mapping[g]["ensembl"] is None]; print("upper-symbol mapped:", len(panel) - len(miss), "| missing:", miss, flush=True)
mm = post("https://rest.ensembl.org/lookup/symbol/mus_musculus", miss) if miss else {}
for g in miss:
    mid = mm.get(g, {}).get("id"); tok = None; how = None
    if mid in vocab: tok, how = mid, "mouse_id_in_vocab"
    elif mid:
        for attempt in range(4):
            rr = S.get(f"https://rest.ensembl.org/homology/id/mus_musculus/{mid}", params={"target_species": "homo_sapiens", "type": "orthologues", "format": "condensed"}, timeout=120)
            if rr.status_code == 200:
                try:
                    homs = [h for h in rr.json().get("data", [{}])[0].get("homologies", []) if h.get("species") == "homo_sapiens"]
                    for h in ([x for x in homs if x.get("type") == "ortholog_one2one"] or homs):
                        if h["id"] in vocab: tok, how = h["id"], h["type"]; break
                except Exception: pass
                break
            time.sleep(2 + 2 * attempt)
    mapping[g] = {"ensembl": tok, "how": how, "mouse_id": mid}; print(f"  {g}: {mid} -> {tok} ({how})", flush=True)
for g in panel: mapping[g]["token"] = vocab.get(mapping[g]["ensembl"]) if mapping[g]["ensembl"] else None
json.dump(mapping, open(OUT, "w"), indent=1)
print(f"final mapped {sum(m['token'] is not None for m in mapping.values())}/{len(panel)}; missing: {[g for g in panel if mapping[g]['token'] is None]}")
