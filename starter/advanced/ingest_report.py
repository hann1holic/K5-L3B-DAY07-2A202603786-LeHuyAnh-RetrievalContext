"""Build a page-grounded text/table/visual index for the UNDP HDR 2023/24 PDF."""
from __future__ import annotations
import argparse, json, os, re, time, sys
from pathlib import Path
from urllib.request import Request, urlopen
import pymupdf

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent
sys.path.insert(0, str(STARTER))
PDF = STARTER / "data" / "HDR_2023-24_UNDP.pdf"
OUT = STARTER / "results" / "advanced"

def load_env():
    from day6_adapter import load_env, day6_root
    load_env(day6_root() / "starter" / ".env")
    load_env(STARTER / ".env")

def chunks_from_pdf(pdf: Path, out: Path, render: bool = True):
    doc = pymupdf.open(pdf)
    records=[]; pages=[]; skipped=[]
    for i, page in enumerate(doc):
        page_no=i+1
        raw=page.get_text("text", sort=True)
        text=re.sub(r"\s+", " ", raw).strip()
        pages.append({"page":page_no,"chars":len(text),"text_present":bool(text)})
        if not text:
            skipped.append({"page":page_no,"reason":"No extractable text layer; decorative/image-only content was not OCRed."})
            continue
        # Preserve page locality; overlapping chunks retain source page and PDF coordinates.
        words=text.split()
        for j in range(0,len(words),190):
            piece=" ".join(words[j:j+220])
            if not piece: continue
            records.append({"id":f"p{page_no:03d}_text_{j//190:02d}","page":page_no,
              "kind":"page_text","bbox":None,"text":piece,
              "source":"UNDP Human Development Report 2023/24, English edition",
              "pdf":str(pdf.name)})
        # Add structured table text where layout detection finds tables, but keep the page text too.
        # Layout detection is expensive on prose pages; run it only on pages
        # whose text signals a printed table (including annex tables).
        try:
            tables=page.find_tables().tables if re.search(r"(?i)\bTABLE\s+[0-9]",raw) else []
        except Exception:
            tables=[]
        has_visual=False
        for ti,table in enumerate(tables):
            try: rows=table.extract()
            except Exception: continue
            if not rows: continue
            flattened=[]
            for ri,row in enumerate(rows):
                cells=[re.sub(r"\s+"," ",str(c or "")).strip() for c in row]
                flattened.append(f"row {ri+1}: " + " | ".join(cells))
            records.append({"id":f"p{page_no:03d}_table_{ti+1}","page":page_no,
              "kind":"table","bbox":list(table.bbox),"text":"TABLE extracted from page layout. " + " ".join(flattened),
              "source":"UNDP Human Development Report 2023/24, English edition","pdf":str(pdf.name),"table_rows":len(rows)})
        # Figure captions give a searchable bridge to charts, maps, and diagrams.
        for m in re.finditer(r"(?i)\b(Figure\s+[A-Z]?\.?\d+(?:\.\d+)?)\s+([^.!?]{12,260})",raw):
            has_visual=True
            cap=re.sub(r"\s+"," ",m.group(0)).strip()
            bbox=None
            try:
                rects=page.search_for(m.group(1))
                if rects: bbox=list(rects[0] | page.search_for(cap.split(" ",2)[-1])[0]) if len(page.search_for(cap.split(" ",2)[-1])) else list(rects[0])
            except Exception: pass
            fid=f"p{page_no:03d}_figure_{len([r for r in records if r['page']==page_no and r['kind']=='figure_caption'])+1:02d}"
            records.append({"id":fid,"page":page_no,"kind":"figure_caption","bbox":bbox,
              "text":cap,"source":"UNDP Human Development Report 2023/24, English edition","pdf":str(pdf.name)})
        if render and (has_visual or tables):
            # Render pages with useful figures/tables; prose pages remain fully indexed as text.
            (out/"pages").mkdir(parents=True,exist_ok=True)
            pix=page.get_pixmap(matrix=pymupdf.Matrix(110/72,110/72),alpha=False)
            pix.save(out/"pages"/f"page-{page_no:03d}.png")
            pages[-1]["visual_crop"] = f"pages/page-{page_no:03d}.png"
    doc.close()
    return records,pages,skipped

def embed(records, api_key, model, base):
    import json as _json
    vectors=[]; total_tokens=0
    for start in range(0,len(records),64):
        batch=records[start:start+64]
        payload=_json.dumps({"model":model,"input":[r["text"] for r in batch]}).encode()
        req=Request(base+"/embeddings",payload,{"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},method="POST")
        with urlopen(req,timeout=180) as response: result=_json.loads(response.read())
        ordered=sorted(result["data"],key=lambda x:x["index"])
        vectors.extend(x["embedding"] for x in ordered)
        total_tokens+=int(result.get("usage",{}).get("total_tokens",0) or 0)
        print(f"embedded {min(start+64,len(records))}/{len(records)} chunks")
    return vectors,total_tokens

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pdf",type=Path,default=PDF); ap.add_argument("--no-render",action="store_true"); ap.add_argument("--rebuild",action="store_true"); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    chunk_path=OUT/"chunks.json"; inventory_path=OUT/"page_inventory.json"
    if chunk_path.exists() and inventory_path.exists() and not args.rebuild:
        records=json.loads(chunk_path.read_text(encoding="utf-8"))
        inv=json.loads(inventory_path.read_text(encoding="utf-8")); pages=inv["pages"]; skipped=inv["skipped"]
        print(f"reusing extracted source inventory: {len(records)} chunks from {len(pages)} pages")
    else:
        records,pages,skipped=chunks_from_pdf(args.pdf,OUT,not args.no_render)
        chunk_path.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
        inventory_path.write_text(json.dumps({"pages":pages,"skipped":skipped},ensure_ascii=False,indent=2),encoding="utf-8")
    # Avoid re-embedding and duplicate API charges when the saved index already
    # matches the extracted records. Use --rebuild only when source/code changes.
    import chromadb
    path=OUT/"chroma"; name="undp_hdr_2023_24"
    if path.exists() and not args.rebuild:
        try:
            existing=chromadb.PersistentClient(path=str(path)).get_collection(name)
            if existing.count()==len(records):
                print(f"index already current ({existing.count()} chunks); use --rebuild to recreate it")
                return
        except Exception:
            pass
    load_env()
    key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
    base=(os.getenv("EMBEDDING_BASE_URL") or ("https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY") else "https://api.openai.com/v1")).rstrip("/")
    if not key: raise RuntimeError("Missing embedding API credential in local Day 6/Day 7 .env")
    vectors,tokens=embed(records,key,model,base)
    client=chromadb.PersistentClient(path=str(path))
    try: client.delete_collection(name)
    except Exception: pass
    col=client.create_collection(name,metadata={"hnsw:space":"cosine"})
    for start in range(0,len(records),256):
        batch=records[start:start+256]
        col.add(ids=[x["id"] for x in batch],embeddings=vectors[start:start+256],documents=[x["text"] for x in batch],
          metadatas=[{"page":x["page"],"kind":x["kind"],"source":x["source"],"pdf":x["pdf"],"bbox":json.dumps(x["bbox"]),"render":f"pages/page-{x['page']:03d}.png"} for x in batch])
    meta={"pdf":args.pdf.name,"pages":len(pages),"chunks":len(records),"page_text_chunks":sum(x["kind"]=="page_text" for x in records),"table_chunks":sum(x["kind"]=="table" for x in records),"figure_caption_chunks":sum(x["kind"]=="figure_caption" for x in records),"pages_without_text":skipped,"embedding_model":model,"embedding_tokens_reported":tokens,"embedding_cost_usd":None,"rendered_pages":sum(bool(p.get("visual_crop")) for p in pages),"index_path":"starter/results/advanced/chroma","collection":name}
    (OUT/"index_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
