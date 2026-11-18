"""FastAPI service. Reference access is confined to the optional evaluation block."""
import json, logging, os, time
from pathlib import Path
from threading import Lock
import joblib
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field,ConfigDict
from core import Geometry,load_trip,load_reference,align_reference,candidate_probabilities,viterbi,stitch_route,direct_continuity

HERE=Path(__file__).parent
DATA_ROOT=Path(os.environ.get('PATHSNAP_DATA',str(HERE.parent/'dataset'/'Student folder')))
MODE=os.environ.get('PATHSNAP_MODE','development')
if MODE not in ('development','evaluator'):raise ValueError('Invalid PATHSNAP_MODE')
PARTITION='student_test' if MODE=='development' else 'evaluator_test'
MODEL_PATH=Path(os.environ.get('PATHSNAP_MODEL',str(HERE/'artifacts'/'final_model.joblib')))
origins=os.environ.get('PATHSNAP_UI_ORIGINS','http://localhost:8090,http://127.0.0.1:8090').split(',')
app=FastAPI(title='PathSnap',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])
logger=logging.getLogger('uvicorn.error');lock=Lock();bundle=None

class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    trip_id:str=Field(pattern=r'^[0-9]{8}$',min_length=8,max_length=8)

def available_trips():
    folder=DATA_ROOT/PARTITION
    if not folder.is_dir():return []
    return sorted(d.name for d in folder.iterdir() if d.is_dir() and d.name.isascii() and len(d.name)==8 and d.name.isdigit() and (d/f'{d.name}.track').is_file())

def get_bundle():
    global bundle
    if bundle is None:
        if not MODEL_PATH.is_file():raise HTTPException(503,'Trained model unavailable; run workflow first')
        bundle=joblib.load(MODEL_PATH)
    return bundle

def infer(trip,model_bundle):
    """Only GPS + graph + trained artifacts. No references accepted or loaded."""
    g=Geometry(trip);c=g.candidates();features=g.features(c)
    p=candidate_probabilities(model_bundle['model'],features,c['valid'])
    sequence,states,resets=viterbi(g,c,p,**model_bundle['decoder'])
    route,breaks=stitch_route(g,sequence)
    return g,sequence,route,breaks,resets

@app.get('/health')
def health():
    return {'status':'ok','mode':MODE,'model_available':MODEL_PATH.is_file(),'trip_count':len(available_trips()),'partition':PARTITION}

@app.get('/trips')
def trips():
    ids=available_trips()
    return {'mode':MODE,'partition':PARTITION,'trips':ids,'reserved_evaluator_verified':False,
            'message': 'Permitted student-test demonstration data' if MODE=='development' else ('Evaluator dataset available' if ids else 'Reserved evaluator dataset not supplied; no development fallback')}

@app.post('/predict')
def predict(request:PredictRequest):
    if request.trip_id not in available_trips():raise HTTPException(404,'Trip not available in the configured partition')
    if not lock.acquire(blocking=False):raise HTTPException(429,'Inference busy; retry after the current trip completes')
    try:
        started=time.perf_counter();trip=load_trip(DATA_ROOT,PARTITION,request.trip_id)
        logger.info('PathSnap inference request mode=%s trip=%s gps=%s',MODE,request.trip_id,len(trip.gps))
        g,sequence,route,breaks,resets=infer(trip,get_bundle())
        inference_seconds=time.perf_counter()-started
        # References are read strictly after inference, used only by this server-side scorer.
        evaluation=None;refpath=DATA_ROOT/PARTITION/request.trip_id/f'{request.trip_id}.route'
        if refpath.is_file():
            reference=load_reference(DATA_ROOT,PARTITION,request.trip_id,len(trip.arcs));aligned=align_reference(g,reference);mask=aligned['confident']
            evaluation={'reference_derived_top1_agreement':float(np.mean(sequence[mask]==aligned['labels'][mask])) if mask.any() else None,
                'scored_observations':int(mask.sum()),'label_coverage':float(mask.mean()),
                'label_basis':'Chronological reference-route-derived weak labels; not independently verified pointwise ground truth'}
        # Clip road context to the observed trip's extent. Geometry only, never reference IDs.
        lo=g.points.min(0)-100;hi=g.points.max(0)+100
        mid=g.a+.5*g.v;near=np.all((mid>=lo)&(mid<=hi),axis=1)
        roads=trip.nodes[trip.arcs[np.flatnonzero(near)]].tolist()
        segments=trip.nodes[trip.arcs[np.asarray(route,dtype=int)]].tolist()
        result={'trip_id':request.trip_id,'mode':MODE,'gps':trip.gps[:,:2].tolist(),'road_segments':roads,
                'predicted_segments':segments,'predicted_arc_ids':sequence.tolist(),'route_break_indices':breaks,
                'summary':{'gps_observations':len(sequence),'reconstructed_arcs':len(route),'decoder_resets':len(resets),
                           'unconnected_stitch_breaks':len(breaks),'direct_continuity':direct_continuity(g,sequence),
                           'inference_seconds':inference_seconds,'total_seconds':time.perf_counter()-started},'evaluation':evaluation}
        logger.info('PathSnap response trip=%s seconds=%.3f reference_labels_exposed=false',request.trip_id,result['summary']['total_seconds'])
        return result
    except FileNotFoundError as e:raise HTTPException(422,'Trip files incomplete') from e
    except ValueError as e:raise HTTPException(422,str(e)) from e
    finally:lock.release()
