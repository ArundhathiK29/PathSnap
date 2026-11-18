"""Actual-artifact audit: reproduce metrics, refit selected model, exercise API."""
import argparse,ast,hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from workflow import OUT,ART,HERE,write_json,student_metrics,model_factory

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',default=str(HERE.parent/'dataset'/'Student folder'));args=parser.parse_args()
    data=joblib.load(HERE/'cache'/'prepared.joblib');bundle=joblib.load(ART/'final_model.joblib');checks=[]
    def check(name,passed,detail):
        checks.append({'check':name,'status':'PASS' if passed else 'FAIL','detail':detail});write_json(OUT/'verification.json',checks);print(checks[-1],flush=True)
    folds=json.loads((OUT/'cv_folds.json').read_text())
    check('five_distinct_models',len(folds)==5,list(folds))
    for name,records in folds.items():
        valid=all(not set(r['train_trips'])&set(r['validation_trips']) for r in records)
        union=set().union(*(set(r['validation_trips']) for r in records))
        check('trip_grouping_'+name,valid and len(union)==80,{'folds':len(records),'validation_trip_union':len(union)})
    metrics,rows=student_metrics(bundle['model'],data,args.data,'Reproduced_final')
    saved=json.loads((OUT/'tuning_summary.json').read_text())['student_metrics']
    keys=['student_macro_f1','top1_reference_agreement','candidate_binary_log_loss']
    check('reproduced_saved_metrics',all(abs(metrics[k]-saved[k])<1e-10 for k in keys),{k:metrics[k] for k in keys})
    # Independently refit actual winning configuration, compare held-out predictions.
    refit=model_factory(bundle['name'],bundle['parameters']);t=time.perf_counter();refit.fit(data['X'],data['y'])
    sample=data['test'][0]['X'].reshape(-1,len(bundle['features']))[:4000]
    check('deterministic_final_refit',np.allclose(refit.predict_proba(sample),bundle['model'].predict_proba(sample),atol=1e-10),{'seconds':time.perf_counter()-t,'held_out_candidate_rows':len(sample)})
    import api
    api.DATA_ROOT=Path(args.data);api.MODE='development';api.PARTITION='student_test';api.bundle=None
    import logging
    handler=logging.FileHandler(OUT/'api_handler.log',mode='w',encoding='utf-8');api.logger.addHandler(handler);api.logger.setLevel(logging.INFO)
    listing=api.trips()
    check('handler_ten_development_trips',len(listing['trips'])==10,listing)
    from pydantic import ValidationError
    rejected=False
    try:api.PredictRequest(trip_id='../etc')
    except ValidationError:rejected=True
    check('invalid_identifier_schema_rejected',rejected,'Pydantic request-schema test; HTTP transport unverified')
    rejected=False
    try:api.predict(api.PredictRequest(trip_id='99999999'))
    except api.HTTPException as e:rejected=e.status_code==404
    check('unknown_trip_handler_rejected',rejected,'Handler raises HTTPException(404); transport unverified')
    responses=[]
    for tid in listing['trips']:
        t=time.perf_counter();result=api.predict(api.PredictRequest(trip_id=tid))
        expected=json.loads((OUT/f'prediction_{tid}.json').read_text())['decoded_arc_ids']
        ok=result.get('predicted_arc_ids')==expected and len(result.get('gps',[]))==len(expected)
        json.dumps(result,allow_nan=False)
        forbidden={'reference','ground_truth','reference_arc_ids','reference_route','labels'}
        def private_free(x):
            if isinstance(x,dict):return not forbidden&x.keys() and all(private_free(v) for v in x.values())
            if isinstance(x,list):return all(private_free(v) for v in x)
            return True
        check('api_handler_inference_'+tid,ok and private_free(result),{'transport':'direct validated Python handler; not HTTP','seconds':time.perf_counter()-t,'observations':len(expected)})
        responses.append({'trip_id':tid,'transport':'direct Python handler','summary':result.get('summary'),'evaluation':result.get('evaluation')})
    repeated=api.predict(api.PredictRequest(trip_id=listing['trips'][0]))
    check('repeated_request_deterministic',repeated['predicted_arc_ids']==json.loads((OUT/f"prediction_{listing['trips'][0]}.json").read_text())['decoded_arc_ids'],'Same request predicts the same sequence')
    api.PARTITION='evaluator_test';api.MODE='evaluator'
    check('reserved_mode_no_development_fallback',api.trips()['trips']==[],'Reserved dataset absent; zero fabricated trip choices')
    api.PARTITION='student_test';api.MODE='development'
    write_json(OUT/'api_integration_responses.json',responses)
    checks.append({'check':'live_http_and_browser','status':'UNVERIFIED','detail':'Windows sandbox blocks sockets, including asyncio socketpair. TestClient could not start; direct handlers and Pydantic schemas tested instead. Browser automation also unavailable.'})
    write_json(OUT/'verification.json',checks)
    manifest=[]
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and not any(x in f.parts for x in ['cache','__pycache__','.pytest_cache','.git']):
            manifest.append({'path':f.relative_to(HERE).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    write_json(OUT/'artifact_manifest.json',manifest)
    print('AUDIT',sum(c['status']=='PASS' for c in checks),'/',len(checks),flush=True)

if __name__=='__main__':main()
