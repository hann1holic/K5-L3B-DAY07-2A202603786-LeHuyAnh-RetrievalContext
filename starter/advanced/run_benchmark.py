"""Run real Day 7 tool-calling agent against the frozen UNDP advanced benchmark."""
import argparse,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; STARTER=HERE.parent; ROOT=STARTER.parent
sys.path.insert(0,str(STARTER))
from day6_adapter import load_env,day6_root,EMBEDDING_STATS
from model_controller import OpenAIController
from agentic import run_agent

class ReportRetriever:
    def __init__(self):
        import chromadb
        p=STARTER/"results"/"advanced"/"chroma"
        self.col=chromadb.PersistentClient(path=str(p)).get_collection("undp_hdr_2023_24")
    def search(self,query,*,filters=None,top_k=3):
        from day6_adapter import embed_query
        result=self.col.query(query_embeddings=[embed_query(query)],n_results=top_k,include=["documents","metadatas","distances"])
        return [{"id":cid,"source_id":"undp_hdr_2023_24","source":m["source"],"page":m["page"],"text":txt,
                 "metadata":m,"score":1-float(d),"strategy":"dense_report"}
                for cid,txt,m,d in zip(result["ids"][0],result["documents"][0],result["metadatas"][0],result["distances"][0])]

def main():
    p=argparse.ArgumentParser();p.add_argument("--split",choices=["dev","test"],required=True);p.add_argument("--top-k",type=int,default=5);p.add_argument("--budget",type=int,default=9000);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    qfile=STARTER/"data"/"advanced_questions.json"; cases=[x for x in json.loads(qfile.read_text(encoding="utf-8")) if x["split"]==a.split]
    if a.split=="test" and len(cases)!=30: raise ValueError(f"Frozen advanced test must have 30 cases; found {len(cases)}")
    load_env(day6_root()/"starter"/".env");load_env(STARTER/".env")
    controller=OpenAIController();retriever=ReportRetriever(); before=dict(EMBEDDING_STATS);traces=[]
    for i,q in enumerate(cases,1):
        start=time.perf_counter()
        try:
            r=run_agent(q["question"],controller,retriever,max_searches=3,top_k=a.top_k,budget=a.budget)
            trace={"query_id":q["id"],"split":a.split,"question":q["question"],**r,"elapsed_seconds":round(time.perf_counter()-start,3)}
        except Exception as e:
            trace={"query_id":q["id"],"split":a.split,"question":q["question"],"status":"error","answer":None,"citations":[],"events":[],"error_type":type(e).__name__,"elapsed_seconds":round(time.perf_counter()-start,3)}
        traces.append(trace);print(f"[{i}/{len(cases)}] {q['id']}: {trace['status']} ({trace['elapsed_seconds']}s)",flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"traces.json").write_text(json.dumps(traces,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.output/"cases.json").write_text(json.dumps([{"id":q["id"],"split":q["split"],"answerable":q["answerable"],"main_category":q["main_category"]} for q in cases],ensure_ascii=False,indent=2),encoding="utf-8")
    meta={"split":a.split,"strategy":"dense_report","top_k":a.top_k,"context_budget_chars":a.budget,"model":controller.model,"model_calls":controller.calls,"prompt_tokens_reported":controller.prompt_tokens,"completion_tokens_reported":controller.completion_tokens,"provider_reported_cost_usd":round(controller.reported_cost_usd,8) if controller.cost_reporting_calls==controller.calls and controller.calls else None,"provider_cost_reporting_calls":controller.cost_reporting_calls,"query_embedding_requests":EMBEDDING_STATS["requests"]-before["requests"],"query_embedding_tokens":EMBEDDING_STATS["tokens"]-before["tokens"],"errors":sum(t["status"]=="error" for t in traces)}
    (a.output/"run_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
