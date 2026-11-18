"""Reproducible exploration, grouped experiments and GP Bayesian optimisation."""
import argparse, hashlib, json, platform, time, warnings
from pathlib import Path
import joblib
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
THREAD_LIMIT=threadpool_limits(limits=2)
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import f1_score,precision_score,recall_score,log_loss,confusion_matrix
from core import *

SEED=2911
HERE=Path(__file__).parent
OUT=HERE/'results'; ART=HERE/'artifacts'; CACHE=HERE/'cache'
for p in (OUT,ART,CACHE,OUT/'figures',OUT/'maps'):p.mkdir(exist_ok=True,parents=True)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def model_factory(name,parameters=None):
    p=parameters or {}
    if name=='LogisticRegression':
        return make_pipeline(StandardScaler(),LogisticRegression(C=p.get('C',1.),max_iter=350,random_state=SEED))
    if name=='GaussianNB':return make_pipeline(StandardScaler(),GaussianNB(var_smoothing=p.get('var_smoothing',1e-6)))
    if name=='DecisionTree':return DecisionTreeClassifier(max_depth=p.get('max_depth',10),min_samples_leaf=p.get('min_samples_leaf',30),random_state=SEED)
    if name=='RandomForest':return RandomForestClassifier(n_estimators=32,max_depth=p.get('max_depth',14),min_samples_leaf=p.get('min_samples_leaf',8),max_features=p.get('max_features',0.8),n_jobs=1,random_state=SEED)
    if name=='HistGradientBoosting':return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=p.get('max_leaf_nodes',23),learning_rate=p.get('learning_rate',0.09),l2_regularization=p.get('l2_regularization',1.),early_stopping=False,random_state=SEED)
    raise ValueError(name)

def draw_map(g,c=None,predicted=None,reference=None,name='trip'):
    """Use genuine Folium when installed; clearly labelled Leaflet fallback otherwise."""
    import sys
    if (HERE/'vendor').is_dir():sys.path.insert(0,str(HERE/'vendor'))
    try:
        import folium
    except ImportError:
        folium=None
    gps=g.trip.gps[:,:2][:,::-1].tolist()
    segments=lambda ids:[g.trip.nodes[g.trip.arcs[i]][:,::-1].tolist() for i in ids]
    road_ids=np.arange(len(g.trip.arcs))
    lo=g.points.min(axis=0)-150;hi=g.points.max(axis=0)+150
    near=np.all((g.a>=lo)&(g.a<=hi),axis=1)|np.all((g.a+g.v>=lo)&(g.a+g.v<=hi),axis=1)
    roads=segments(road_ids[near])
    ref=segments(reference) if reference is not None else []
    pred=segments(np.unique(predicted)) if predicted is not None else []
    html_path=OUT/'maps'/f'{name}.html'
    if folium:
        m=folium.Map(location=np.mean(gps,axis=0).tolist(),zoom_start=13)
        for label,lines,color,weight in [('Road network',roads,'#94a3b8',1),('Reference route',ref,'#16a34a',5),('Predicted road segments',pred,'#f97316',3)]:
            layer=folium.FeatureGroup(name=label)
            for line in lines:folium.PolyLine(line,color=color,weight=weight,opacity=0.8).add_to(layer)
            layer.add_to(m)
        layer=folium.FeatureGroup(name='GPS observations')
        for point in gps:folium.CircleMarker(point,radius=2,color='#2563eb',fill=True).add_to(layer)
        layer.add_to(m);folium.LayerControl().add_to(m)
        legend='<div style="position:fixed;bottom:30px;left:30px;background:white;padding:12px;z-index:9999">Blue: GPS | Grey: roads | Green: reference | Orange: prediction</div>'
        m.get_root().html.add_child(folium.Element(legend));m.fit_bounds([np.min(gps,axis=0).tolist(),np.max(gps,axis=0).tolist()]);m.save(str(html_path))
        kind='folium'
    else:
        # Fallback keeps application usable; it does NOT satisfy mandatory Folium execution.
        data=json.dumps({'gps':gps,'roads':roads,'reference':ref,'predicted':pred})
        html='''<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>html,body,#map{height:100%;margin:0}.legend{background:white;padding:10px}</style></head><body><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>const D=DATA;const m=L.map('map');L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors'}).addTo(m);const layers={};for(const [key,color,width] of [['roads','#94a3b8',1],['reference','#16a34a',5],['predicted','#f97316',3]]){layers[key]=L.layerGroup(D[key].map(x=>L.polyline(x,{color,weight:width}))).addTo(m)}layers.gps=L.layerGroup(D.gps.map(x=>L.circleMarker(x,{radius:2,color:'#2563eb'}))).addTo(m);L.control.layers(null,layers).addTo(m);m.fitBounds(D.gps);const legend=L.control({position:'bottomleft'});legend.onAdd=()=>{const d=L.DomUtil.create('div','legend');d.textContent='Blue: GPS | Grey: roads | Green: reference | Orange: prediction. Leaflet fallback; Folium unavailable.';return d};legend.addTo(m);</script></body></html>'''.replace('DATA',data)
        html_path.write_text(html,encoding='utf-8');kind='leaflet_fallback'
    return kind

def plot_trip(g,reference=None,predicted=None,name='trip'):
    fig,ax=plt.subplots(figsize=(9,6))
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.stack([g.a,g.a+g.v],axis=1),colors='#cbd5e1',linewidths=.5))
    if reference is not None:ax.add_collection(LineCollection(np.stack([g.a[reference],g.a[reference]+g.v[reference]],axis=1),colors='#16a34a',linewidths=3,label='Reference'))
    if predicted is not None:ax.add_collection(LineCollection(np.stack([g.a[predicted],g.a[predicted]+g.v[predicted]],axis=1),colors='#f97316',linewidths=1.4,label='Prediction'))
    ax.scatter(*g.points.T,s=3,color='#2563eb',label='GPS')
    lo=g.points.min(0)-100;hi=g.points.max(0)+100
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Local east (m)',ylabel='Local north (m)',title=f'PathSnap trip {g.trip.trip_id}')
    ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(OUT/'figures'/f'{name}.png',dpi=150);plt.close(fig)

def prepare(root,sample_per_trip=240):
    inventories=[];train=[];testing=[];rng=np.random.default_rng(SEED)
    for part in ('train','student_test'):
        for folder in sorted((Path(root)/part).iterdir()):
            if not folder.is_dir():continue
            started=time.perf_counter();trip=load_trip(root,part,folder.name);g=Geometry(trip)
            route=load_reference(root,part,folder.name,len(trip.arcs))
            labels=align_reference(g,route);c=g.candidates();X=g.features(c)
            hit=np.any((c['ids']==labels['labels'][:,None])&c['valid'],axis=1)
            y=(c['ids']==labels['labels'][:,None]).astype(np.uint8)
            conf=labels['confident'];eligible=np.flatnonzero(conf&hit)
            selected=np.sort(rng.choice(eligible,size=min(sample_per_trip,len(eligible)),replace=False))
            nweak=connected_components(g.graph,directed=True,connection='weak',return_labels=False)
            nstrong=connected_components(g.graph,directed=True,connection='strong',return_labels=False)
            discontinuities=int(np.sum(trip.arcs[route[:-1],1]!=trip.arcs[route[1:],0]))
            inventory={'partition':part,'trip_id':folder.name,'gps_count':len(trip.gps),'nodes':len(trip.nodes),'arcs':len(trip.arcs),
                'route_entries':len(route),'gps_missing':int(np.isnan(trip.gps).sum()),'gaps_gt_1s':int(np.sum(g.dt[1:]>1.001)),
                'max_gap_s':float(g.dt.max()),'duplicate_coordinates':len(g.points)-len(np.unique(g.points,axis=0)),
                'weak_components':int(nweak),'strong_components':int(nstrong),'reference_discontinuities':discontinuities,
                'zero_length_arcs':int(np.sum(g.length<.01)),'confident_labels':int(conf.sum()),'confidence_coverage':float(conf.mean()),
                'candidate_recall':float(hit[conf].mean()) if conf.any() else None,'selected_observations':len(selected),
                'alignment_median_distance_m':float(np.median(labels['distance'])),'alignment_resets':int(labels['resets'].sum()),
                'lon_min':float(trip.gps[:,0].min()),'lon_max':float(trip.gps[:,0].max()),'lat_min':float(trip.gps[:,1].min()),'lat_max':float(trip.gps[:,1].max()),
                'processing_seconds':time.perf_counter()-started}
            inventories.append(inventory)
            if part=='train':
                train.append({'X':X[selected].reshape(-1,len(FEATURES)),'y':y[selected].ravel(),
                    'groups':np.full(len(selected)*X.shape[1],int(folder.name)), 'observation_ids':selected,
                    'trip_id':folder.name})
            else:
                item={'trip_id':folder.name,'X':X,'y':y,'candidate':c,'alignment':labels,'reference':route}
                testing.append(item)
            if len(inventories)==1:
                plot_trip(g,route,name='eda_example');map_kind=draw_map(g,reference=route,name='eda_example')
                write_json(OUT/'map_backend.json',{'kind':map_kind})
            print('PREPARE',part,folder.name,len(trip.gps),'confident',round(conf.mean(),3),'candidate_recall',inventory['candidate_recall'],flush=True)
    pd.DataFrame(inventories).to_csv(OUT/'inventory.csv',index=False)
    data={'X':np.concatenate([t['X'] for t in train]),'y':np.concatenate([t['y'] for t in train]),'groups':np.concatenate([t['groups'] for t in train]),'test':testing,'sample_per_trip':sample_per_trip}
    joblib.dump(data,CACHE/'prepared.joblib',compress=3)
    write_json(OUT/'data_summary.json',{'partitions':pd.DataFrame(inventories).groupby('partition')[['gps_count','confident_labels','gaps_gt_1s','reference_discontinuities']].sum().to_dict(),
        'training_candidate_rows':len(data['y']),'training_positive_rate':float(data['y'].mean()),'features':FEATURES,'seed':SEED,'sample_per_trip':sample_per_trip,
        'label_warning':'All observation-level labels are reference-derived weak labels, not independent ground truth.'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.4));df=pd.DataFrame(inventories)
    for ax,key,title in zip(axes,['gps_count','max_gap_s','confidence_coverage'],['GPS count per trip','Maximum sampling gap (seconds)','Reference-label confidence coverage']):
        for part,color in [('train','#2563eb'),('student_test','#f97316')]:ax.hist(df[df.partition==part][key],bins=15,alpha=.65,label=part,color=color)
        ax.set_title(title,fontsize=10);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'figures'/'eda_summary.png',dpi=150);plt.close(fig)
    return data

def cv_evaluate(name,data,parameters=None):
    X,y,groups=data['X'],data['y'],data['groups'];records=[]
    for fold,(a,b) in enumerate(GroupKFold(n_splits=3).split(X,y,groups),1):
        model=model_factory(name,parameters);t=time.perf_counter();model.fit(X[a],y[a]);train_seconds=time.perf_counter()-t
        pred=model.predict(X[b]);tr=model.predict(X[a])
        records.append({'fold':fold,'macro_f1':float(f1_score(y[b],pred,average='macro',zero_division=0)),
            'train_macro_f1':float(f1_score(y[a],tr,average='macro',zero_division=0)), 'training_seconds':train_seconds,
            'train_trips':[f'{x:08d}' for x in np.unique(groups[a])], 'validation_trips':[f'{x:08d}' for x in np.unique(groups[b])]})
    return float(np.mean([r['macro_f1'] for r in records])),records

def student_metrics(model,data,root,name):
    ys=[];ps=[];preds=[];rows=[];total_time=0
    for item in data['test']:
        X=item['X'];c=item['candidate'];a=item['alignment'];valid_obs=a['confident']; y=item['y']
        t=time.perf_counter();binary=model.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(X.shape[:2]);elapsed=time.perf_counter()-t;total_time+=elapsed
        p=np.where(c['valid'],np.clip(binary,1e-8,1-1e-8),0);p/=np.maximum(p.sum(1,keepdims=True),1e-12)
        predicted=c['ids'][np.arange(len(p)),np.argmax(p,axis=1)]
        hit=np.any(y.astype(bool)&c['valid'],axis=1)
        yflat=y[valid_obs].ravel();pflat=binary[valid_obs].ravel();ys.extend(yflat);ps.extend(pflat);preds.extend(pflat>=.5)
        g=Geometry(load_trip(root,'student_test',item['trip_id']))
        correct=(predicted==a['labels'])&valid_obs
        prob_ref=np.sum(p*y,axis=1)
        rows.append({'model':name,'trip_id':item['trip_id'],'observations':len(p),'scored_observations':int(valid_obs.sum()),
            'correct':int(correct.sum()),'top1_reference_agreement':float(correct.sum()/max(1,valid_obs.sum())),
            'all_aligned_top1_agreement':float(np.mean(predicted==a['labels'])),
            'candidate_miss_rate':float(1-hit[valid_obs].mean()),'direct_route_continuity':direct_continuity(g,predicted),
            'observation_log_loss':float(-np.log(np.clip(prob_ref[valid_obs],1e-12,1)).mean()),'inference_seconds':elapsed})
    ys=np.array(ys);ps=np.array(ps);preds=np.array(preds)
    metrics={'student_macro_f1':float(f1_score(ys,preds,average='macro',zero_division=0)),
        'precision':float(precision_score(ys,preds,zero_division=0)),'recall':float(recall_score(ys,preds,zero_division=0)),
        'candidate_binary_log_loss':float(log_loss(ys,ps,labels=[0,1])),
        'top1_reference_agreement':sum(r['correct'] for r in rows)/sum(r['scored_observations'] for r in rows),
        'candidate_miss_rate':float(np.mean([r['candidate_miss_rate'] for r in rows])),
        'direct_route_continuity':float(np.mean([r['direct_route_continuity'] for r in rows])),
        'inference_seconds':total_time,'scored_observations':int(sum(r['scored_observations'] for r in rows))}
    write_json(OUT/f'{name}_confusion.json',confusion_matrix(ys,preds,labels=[0,1]).tolist())
    return metrics,rows

def experiment(root,data):
    records=pd.read_csv(OUT/'model_metrics.csv').to_dict('records') if (OUT/'model_metrics.csv').exists() else []
    pertrip=pd.read_csv(OUT/'model_per_trip.csv',dtype={'trip_id':str}).to_dict('records') if (OUT/'model_per_trip.csv').exists() else []
    folds=json.loads((OUT/'cv_folds.json').read_text()) if (OUT/'cv_folds.json').exists() else {}
    for name in ['LogisticRegression','GaussianNB','DecisionTree','RandomForest','HistGradientBoosting']:
        if any(r['model']==name for r in records):
            if not any(r['model']==name for r in pertrip):
                _,rows=student_metrics(joblib.load(ART/f'{name}.joblib'),data,root,name);pertrip.extend(rows)
            continue
        print('TRAIN',name,flush=True);cv,fold=cv_evaluate(name,data);folds[name]=fold
        model=model_factory(name);started=time.perf_counter();model.fit(data['X'],data['y']);elapsed=time.perf_counter()-started
        metrics,rows=student_metrics(model,data,root,name);pertrip.extend(rows)
        records.append({'model':name,'cv_macro_f1':cv,'train_macro_f1':float(f1_score(data['y'],model.predict(data['X']),average='macro')), 'training_seconds':elapsed,**metrics})
        joblib.dump(model,ART/f'{name}.joblib',compress=3)
        write_json(OUT/f'{name}_configuration.json',{k:v for k,v in model.get_params().items() if isinstance(v,(str,int,float,bool,type(None)))})
        pd.DataFrame(records).to_csv(OUT/'model_metrics.csv',index=False);write_json(OUT/'cv_folds.json',folds)
        pd.DataFrame(pertrip).to_csv(OUT/'model_per_trip.csv',index=False)
        print('RESULT',records[-1],flush=True)
    pd.DataFrame(pertrip).to_csv(OUT/'model_per_trip.csv',index=False)
    # Choose strictly by grouped training CV; student-test stays development diagnostic.
    best=max(records,key=lambda r:r['cv_macro_f1'])['model']
    write_json(OUT/'selection.json',{'model':best,'criterion':'maximum mean trip-grouped CV candidate binary Macro F1','seed':SEED})
    plot_comparison(pd.DataFrame(records))
    return best

def plot_comparison(df):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title in zip(axes.flat,['cv_macro_f1','top1_reference_agreement','student_macro_f1','candidate_binary_log_loss'],['Trip-grouped CV Macro F1','Student-test reference-derived Top-1 agreement','Student-test candidate Macro F1','Student-test candidate log loss']):
        ax.bar(df.model,df[key],color=['#2563eb','#0891b2','#16a34a','#ea580c','#9333ea'][:len(df)])
        ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=25,labelsize=7);ax.set_ylim(bottom=0)
    fig.tight_layout();fig.savefig(OUT/'figures'/'model_comparison.png',dpi=160);plt.close(fig)

def decode_parameters(name,x):
    if name=='RandomForest':return {'max_depth':int(round(7+15*x[0])),'min_samples_leaf':int(round(2+24*x[1])),'max_features':float(.4+.6*x[2])}
    if name=='HistGradientBoosting':return {'max_leaf_nodes':int(round(7+40*x[0])),'learning_rate':float(10**(-1.7+1.1*x[1])),'l2_regularization':float(10**(-2+4*x[2]))}
    if name=='DecisionTree':return {'max_depth':int(round(5+16*x[0])),'min_samples_leaf':int(round(3+60*x[1]))}
    if name=='LogisticRegression':return {'C':float(10**(-3+5*x[0]))}
    return {'var_smoothing':float(10**(-10+8*x[0]))}

def optimise(root,data,trials=10):
    name=json.loads((OUT/'selection.json').read_text())['model'];rng=np.random.default_rng(SEED+1)
    xs=[];scores=[];logs=[];start=time.perf_counter()
    for i in range(trials):
        if i<3:x=rng.random(3);strategy='initial_random_design'
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=np.ones(3),nu=2.5),alpha=1e-4,normalize_y=True,random_state=SEED,n_restarts_optimizer=1)
            with warnings.catch_warnings():warnings.simplefilter('ignore');gp.fit(np.array(xs),np.array(scores))
            pool=rng.random((256,3));mu,sigma=gp.predict(pool,return_std=True);improvement=mu-max(scores)-.0005
            z=improvement/np.maximum(sigma,1e-12);ei=improvement*norm.cdf(z)+sigma*norm.pdf(z)
            x=pool[np.argmax(ei)];strategy='gaussian_process_expected_improvement'
        params=decode_parameters(name,x);t=time.perf_counter();score,folds=cv_evaluate(name,data,params)
        xs.append(x.tolist());scores.append(score);logs.append({'trial':i+1,'strategy':strategy,'unit_coordinates':x.tolist(),'parameters':params,'cv_macro_f1':score,'seconds':time.perf_counter()-t,'fold_results':folds})
        write_json(OUT/'bayesian_trials.json',logs);print('BAYES',i+1,name,params,score,flush=True)
    best=logs[int(np.argmax(scores))];model=model_factory(name,best['parameters']);model.fit(data['X'],data['y'])
    metrics,rows=student_metrics(model,data,root,'Tuned_'+name)
    joblib.dump({'model':model,'name':name,'features':FEATURES,'parameters':best['parameters'],'seed':SEED,'decoder':{'beta':35.,'gap_reset':30.}},ART/'final_model.joblib',compress=3)
    write_json(OUT/'tuning_summary.json',{'model':name,'trials':trials,'best':best,'student_metrics':metrics,'total_seconds':time.perf_counter()-start,'search_space':'decode_parameters() maps unit cube to the documented parameter ranges'})
    pd.DataFrame(rows).to_csv(OUT/'tuned_per_trip.csv',index=False)
    return model

def decoder_experiments(root,data):
    bundle=joblib.load(ART/'final_model.joblib');model=bundle['model'];records=[]
    for item in data['test']:
        tid=item['trip_id'];g=Geometry(load_trip(root,'student_test',tid));c=item['candidate'];p=candidate_probabilities(model,item['X'],c['valid'])
        raw=c['ids'][np.arange(len(p)),p.argmax(1)];t=time.perf_counter();decoded,state,resets=viterbi(g,c,p,**bundle['decoder']);elapsed=time.perf_counter()-t
        aligned=item['alignment'];mask=aligned['confident'];reference=aligned['labels']
        route,breaks=stitch_route(g,decoded)
        ref_arcs=set(map(int,item['reference']));pred_arcs=set(route);inter=len(ref_arcs&pred_arcs)
        records.append({'trip_id':tid,'scored_observations':int(mask.sum()),'classifier_correct':int(np.sum((raw==reference)&mask)),
            'decoder_correct':int(np.sum((decoded==reference)&mask)),'classifier_top1_agreement':float(np.mean(raw[mask]==reference[mask])),
            'decoder_top1_agreement':float(np.mean(decoded[mask]==reference[mask])),
            'classifier_direct_continuity':direct_continuity(g,raw),'decoder_direct_continuity':direct_continuity(g,decoded),
            'stitched_arc_set_precision':inter/max(1,len(pred_arcs)),'stitched_arc_set_recall':inter/max(1,len(ref_arcs)),
            'stitched_route_arc_count':len(route),'unconnected_stitch_breaks':len(breaks),'decoder_resets':len(resets),'decoder_seconds':elapsed,
            'ordered_arc_matching_ratio':SequenceMatcher(None,list(map(int,item['reference'])),route,autojunk=False).ratio()})
        write_json(OUT/f'prediction_{tid}.json',{'trip_id':tid,'classifier_arc_ids':raw.tolist(),'decoded_arc_ids':decoded.tolist(),'stitched_arc_ids':route,'breaks':breaks,'resets':resets})
        pd.DataFrame(records).to_csv(OUT/'decoder_metrics.csv',index=False)
        if len(records)==1:
            plot_trip(g,item['reference'],decoded,name='prediction_example');draw_map(g,c,decoded,item['reference'],name='prediction_example')
        print('DECODE',records[-1],flush=True)
    return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','optimise','decode','all']);parser.add_argument('--data',default=str(HERE.parent/'dataset'/'Student folder'));parser.add_argument('--sample-per-trip',type=int,default=240);parser.add_argument('--trials',type=int,default=10);args=parser.parse_args()
    import sklearn,scipy
    write_json(OUT/'environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__,'platform':platform.platform(),'seed':SEED})
    if args.stage in ('prepare','all'):data=prepare(args.data,args.sample_per_trip)
    else:data=joblib.load(CACHE/'prepared.joblib')
    if args.stage in ('train','all'):experiment(args.data,data)
    if args.stage in ('optimise','all'):optimise(args.data,data,args.trials)
    if args.stage in ('decode','all'):decoder_experiments(args.data,data)
    write_json(HERE/'PROJECT_STATE.json',{'last_completed_command':args.stage,'data_root':args.data,'sample_per_trip':data['sample_per_trip'],'reserved_evaluator_trips':'not supplied','physical_two_laptop':'unverified','github':'not published'})

if __name__=='__main__':main()
