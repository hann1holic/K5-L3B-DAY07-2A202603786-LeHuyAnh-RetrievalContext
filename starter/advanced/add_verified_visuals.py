"""Add human-checked, source-linked descriptions for charts, diagrams, and dense tables."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;STARTER=HERE.parent;ROOT=STARTER.parent;sys.path.insert(0,str(STARTER))
from day6_adapter import load_env,day6_root
SRC="UNDP Human Development Report 2023/24, English edition"
ITEMS=[
 ("p071_fig29_visual",71,"visual_description","Figure 2.9 box plot: planetary pressures index (2022) by HDI group. The median (central line) rises at each group: Low is about 0.01, Medium about 0.03, High about 0.05, and Very high about 0.12. Thus the Very high HDI group has the highest median. Source crop: physical PDF page 71, Figure 2.9."),
 ("p071_fig210_visual",71,"visual_description","Figure 2.10 plots Human Development Index value on the horizontal axis and planetary-pressure index on the vertical axis. The curves labelled 1990 and 2022 show lower average planetary pressure in 2022 for a given HDI level. Source crop: physical PDF page 71, Figure 2.10."),
 ("p029_figo3_visual",29,"visual_description","Figure O.3 is an area chart of the sum of global exports and imports as a share of global GDP. It rises from about 25% in 1970 to around 60% in 2008; after 2008 it fluctuates near 50–60%, ending near 60% in 2022. The periods are labelled Globalization post–Bretton Woods, Hyperglobalization, and A new era? Source crop: physical PDF page 29, Figure O.3."),
 ("p052_fig19_visual",52,"visual_description","Figure 1.9 has two box plots across Low, Medium, High and Very high HDI groups. Left: 2022 planetary-pressure index, increasing with HDI group. Right: projected additional extreme-temperature days above the 1986–2005 average by end-century, decreasing with HDI group; very-high-HDI median is below zero while low-HDI median is about 50 days. Source crop: physical PDF page 52, Figure 1.9."),
 ("p161_fig51_visual",161,"visual_description","Figure 5.1 scatter plot: horizontal axis is percent thinking positively about having a democratic system; vertical axis is percent thinking positively about leaders who may undermine democracy. Periods progress from 1999–2004 through 2017–2022; the latest point is highest on the vertical axis at about 52%. Source crop: physical PDF page 161, Figure 5.1."),
 ("p018_figs12_visual",18,"visual_description","Figure S.1 shows the global HDI trajectory after 2020 below the pre-2019 trend. Figure S.2 shows unequal recovery: around 49% of least-developed countries had recovered their 2019 HDI by 2023, versus all OECD countries shown. Source crop: physical PDF page 18, Figures S.1 and S.2."),
 ("p096_fig31_visual",96,"visual_description","Figure 3.1 vaccine timeline: generic-vaccine authorization/licensing path totals 10.3 years, including 8.8 years until licensure; Pfizer/BioNTech Covid-19 vaccine total is 1.3 years, AstraZeneca/Oxford is over 1.6 years, and Novavax is over 1.4 years. Source crop: physical PDF page 96, Figure 3.1."),
 ("p124_fig42_visual",124,"visual_description","Figure 4.2 is a world map of behavioural-insights units and organizations informing public policy. Its legend classifies efforts as institutions inside government, institutions outside government, and multinational organizations. Source crop: physical PDF page 124, Figure 4.2."),
 ("p124_fig42_uk_agencies",124,"visual_description","Figure 4.2 world map labels several United Kingdom government behavioural-insights organizations, including BEIS (UK), HMCTS (UK), FCDO (UK), HMRC (UK), DfT (UK), DfE (UK), Ofgem (UK), and BIT (UK). Source crop: physical PDF page 124, Figure 4.2; labels visually checked."),
 ("p116_fig41_visual",116,"visual_description","Figure 4.1 diagram: behavioural change and institutional/policy change are the two nodes. Two curved arrows point in opposite directions to show mutual influence. The central text says the interactions are mutually dependent and based on public reasoning and procedures of social choice. Source crop: physical PDF page 116, Figure 4.1."),
 ("p091_table31_verified",91,"table_description","Table 3.1, Recommendations for improving provision of different types of global public goods. Best-shot (example: scientific breakthrough): likely provided if incentives are strong enough for the best shooter to contribute; rich or dominant country fosters provision, but multilateral institutions can pool actions or coordinate multiple potential best shooters. Summation (example: climate change mitigation): tendency toward underprovision from free/easy riding because contributions are perfectly substitutable; grants and loans foster fairness and multilateral institutions monitor and track contributions. Weakest-link (example: pandemic disease control): more likely provided if interests align; capacity building is essential to enhance contributions of the least able and to redistribute capacity among countries. Source crop: physical PDF page 91, Table 3.1."),
 ("p288_table1_switzerland_norway",288,"table_description","Table 1, 2022 rows: Switzerland rank 1, HDI 0.967, life expectancy 84.3 years, expected schooling 16.6 years, mean schooling 13.9 years, GNI per capita $69,433 (2017 PPP); Norway rank 2, HDI 0.966, life expectancy 83.4 years, expected schooling 18.6 years, mean schooling 13.1 years, GNI per capita $69,190 (2017 PPP). Source crop: physical PDF page 288, Table 1."),
 ("p289_table1_malaysia_thailand",289,"table_description","Table 1 rows: Malaysia rank 63, HDI 0.807, life expectancy 76.3 years; Thailand rank 66, HDI 0.803, life expectancy 79.7 years. Source crop: physical PDF page 289, Table 1."),
 ("p315_table7_switzerland",315,"table_description","Table 7 row for Switzerland (2022 HDI, 2021 planetary indicators): HDI 0.967; planetary-pressures-adjusted HDI (PHDI) 0.826; difference from HDI value −14.6%; difference from HDI rank −6; planetary pressure adjustment factor 0.854. Source crop: physical PDF page 315, Table 7."),
 ("p315_table7_norway",315,"table_description","Table 7 row for Norway (2022 HDI, 2021 planetary indicators): HDI 0.966; planetary-pressures-adjusted HDI (PHDI) 0.808; difference from HDI value −16.4%; difference from HDI rank −12; planetary pressure adjustment factor 0.837. Source crop: physical PDF page 315, Table 7."),
]
def main():
 load_env(day6_root()/"starter"/".env");load_env(STARTER/".env")
 key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY");model=os.getenv("EMBEDDING_MODEL","text-embedding-3-small")
 base=(os.getenv("EMBEDDING_BASE_URL") or ("https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY") else "https://api.openai.com/v1")).rstrip("/")
 from urllib.request import Request,urlopen
 records=[{"id":i,"page":p,"kind":k,"text":t,"source":SRC,"pdf":"HDR_2023-24_UNDP.pdf"} for i,p,k,t in ITEMS]
 import chromadb
 col=chromadb.PersistentClient(path=str(STARTER/"results"/"advanced"/"chroma")).get_collection("undp_hdr_2023_24")
 existing=set(col.get(include=[])['ids']); records=[r for r in records if r['id'] not in existing]
 if not records:
  print("verified descriptions already indexed");return
 vectors=[];tokens=0
 for s in range(0,len(records),32):
  batch=records[s:s+32];body=json.dumps({"model":model,"input":[x["text"] for x in batch]}).encode()
  req=Request(base+"/embeddings",body,{"Authorization":f"Bearer {key}","Content-Type":"application/json"},method="POST")
  with urlopen(req,timeout=120) as r: payload=json.loads(r.read())
  vectors += [v["embedding"] for v in sorted(payload["data"],key=lambda x:x["index"])]
  tokens+=int(payload.get("usage",{}).get("total_tokens",0) or 0)
 for s in range(0,len(records),128):
  b=records[s:s+128];col.add(ids=[x["id"] for x in b],embeddings=vectors[s:s+128],documents=[x["text"] for x in b],metadatas=[{"page":x["page"],"kind":x["kind"],"source":SRC,"pdf":x["pdf"],"bbox":"verified visual/table region","render":f"pages/page-{x['page']:03d}.png"} for x in b])
 chunks=STARTER/"results"/"advanced"/"chunks.json";data=json.loads(chunks.read_text(encoding="utf-8"));old={x['id'] for x in data};data.extend(x for x in records if x['id'] not in old);chunks.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
 reviewed=[x for x in data if x['kind'] in {'visual_description','table_description'}]
 out=STARTER/"results"/"advanced"/"visual_review.json";out.write_text(json.dumps({"method":"Human visual verification of source crops; descriptions are search aids, not replacements for original evidence","descriptions":reviewed,"count":len(reviewed),"latest_embedding_tokens_reported":tokens,"embedding_cost_usd":None},ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps({"added_visual_or_table_chunks":len(records),"embedding_tokens_reported":tokens},ensure_ascii=False))
if __name__=="__main__":main()
