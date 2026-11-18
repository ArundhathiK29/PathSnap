"""Geometry, reference alignment, features and directed-network sequence decoding."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.special import logsumexp

FEATURES = ['distance_m','distance_rank','projection_fraction','heading_cosine',
            'segment_length_m','gps_speed_mps','time_gap_s','displacement_m',
            'previous_distance_m','next_distance_m','in_degree','out_degree',
            'heading_available','distance_over_best','speed_heading_interaction']

def read_array(path, columns, integer=False):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError(f'Invalid numeric shape/values: {path}')
    if integer and not np.equal(a, np.floor(a)).all():
        raise ValueError(f'Nonintegral graph index: {path}')
    return a.astype(np.int64) if integer else a

@dataclass
class Trip:
    trip_id: str
    partition: str
    gps: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

def load_trip(root, partition, trip_id):
    if partition not in ('train','student_test','evaluator_test'):
        raise ValueError('Unknown partition')
    if not isinstance(trip_id, str) or len(trip_id)!=8 or not trip_id.isascii() or not trip_id.isdigit():
        raise ValueError('Trip ID must contain eight ASCII digits')
    folder = Path(root)/partition/trip_id
    gps=read_array(folder/f'{trip_id}.track',3)
    nodes=read_array(folder/f'{trip_id}.nodes',2)
    arcs=read_array(folder/f'{trip_id}.arcs',2,True)
    if len(gps)==0 or len(arcs)==0 or np.any(arcs<0) or np.any(arcs>=len(nodes)):
        raise ValueError('Empty trip or invalid node index')
    if np.any(np.diff(gps[:,2])<=0):
        raise ValueError('Timestamps must be strictly increasing; inspect duplicates explicitly')
    if np.any(np.abs(gps[:,:2]) > [180,90]) or np.any(np.abs(nodes)>[180,90]):
        raise ValueError('Implausible longitude/latitude')
    return Trip(trip_id,partition,gps,nodes,arcs)

def load_reference(root, partition, trip_id, arc_count):
    a=read_array(Path(root)/partition/trip_id/f'{trip_id}.route',1,True).ravel()
    if np.any(a<0) or np.any(a>=arc_count):
        raise ValueError('Reference arc outside graph')
    return a

class Geometry:
    def __init__(self, trip):
        self.trip=trip
        self.origin=np.mean(trip.nodes,axis=0)
        self.scale=np.array([np.cos(np.deg2rad(self.origin[1])),1.])*6371008.8*np.pi/180
        self.nodes=self.project(trip.nodes)
        self.points=self.project(trip.gps[:,:2])
        self.a=self.nodes[trip.arcs[:,0]]
        self.v=self.nodes[trip.arcs[:,1]]-self.a
        self.length=np.linalg.norm(self.v,axis=1)
        self.length2=np.maximum(self.length**2,1e-12)
        self.unit=self.v/np.maximum(self.length[:,None],1e-6)
        # CSR sums duplicate entries; that would incorrectly multiply road lengths.
        unique_edges,unique_index=np.unique(trip.arcs,axis=0,return_index=True)
        self.graph=csr_matrix((np.maximum(self.length[unique_index],0.01),unique_edges.T),shape=(len(self.nodes),len(self.nodes)))
        self.outdeg=np.bincount(trip.arcs[:,0],minlength=len(self.nodes))
        self.indeg=np.bincount(trip.arcs[:,1],minlength=len(self.nodes))
        self.dt=np.r_[1.,np.diff(trip.gps[:,2])]
        # Centre displacement reduces jitter; whole-trip offline inference is permitted.
        move=np.empty_like(self.points)
        move[1:-1]=self.points[2:]-self.points[:-2]
        move[0]=self.points[min(1,len(move)-1)]-self.points[0]
        move[-1]=self.points[-1]-self.points[max(0,len(move)-2)]
        norm=np.linalg.norm(move,axis=1)
        self.heading=move/np.maximum(norm[:,None],1e-6)
        self.heading_ok=norm>4
        self.displacement=np.r_[0.,np.linalg.norm(np.diff(self.points,axis=0),axis=1)]
        self.speed=np.minimum(self.displacement/np.maximum(self.dt,0.01),100.)

    def project(self, coordinates):
        return (coordinates-self.origin)*self.scale

    def projection(self, points, ids):
        v=self.v[ids]; a=self.a[ids]
        frac=np.clip(np.sum((points[:,None,:]-a)*v,axis=2)/self.length2[ids],0,1)
        distance=np.linalg.norm(points[:,None,:]-(a+frac[:,:,None]*v),axis=2)
        return distance,frac

    def candidates(self, k=8, arc_pool=None):
        # Sample all segments at <=25m spacing, then exactly rerank retrieved segments.
        pool=np.arange(len(self.a)) if arc_pool is None else np.asarray(arc_pool)
        lengths=self.length[pool]
        n=np.maximum(2,np.ceil(lengths/25).astype(int)+1)
        owners=np.repeat(np.arange(len(pool)),n)
        starts=np.repeat(np.cumsum(n)-n,n)
        frac=(np.arange(n.sum())-starts)/(np.repeat(n,n)-1)
        samples=self.a[pool[owners]]+self.v[pool[owners]]*frac[:,None]
        tree=cKDTree(samples)
        nq=min(48,len(samples))
        near=owners[tree.query(self.points,k=nq,workers=1)[1]]
        if near.ndim==1: near=near[:,None]
        mids=cKDTree(self.a[pool]+0.5*self.v[pool]).query(self.points,k=min(24,len(pool)),workers=1)[1]
        if mids.ndim==1: mids=mids[:,None]
        local=np.sort(np.concatenate([near,mids],axis=1),axis=1)
        duplicate=np.c_[np.zeros(len(local),bool),local[:,1:]==local[:,:-1]]
        ids=pool[local]
        dist,fr=self.projection(self.points,ids)
        dist[duplicate]=np.inf
        order=np.argsort(dist,axis=1,kind='stable')[:,:min(k,len(pool))]
        ids=np.take_along_axis(ids,order,1)
        dist=np.take_along_axis(dist,order,1)
        fr=np.take_along_axis(fr,order,1)
        valid=np.isfinite(dist)
        return {'ids':ids,'distance':np.where(valid,dist,1e6),'fraction':fr,'valid':valid,
                'pool_position':np.take_along_axis(local,order,1)}

    def features(self,c):
        ids=c['ids']; d=c['distance']; f=c['fraction']; n,k=ids.shape
        prev=np.vstack([self.points[:1],self.points[:-1]])
        nxt=np.vstack([self.points[1:],self.points[-1:]])
        dp,_=self.projection(prev,ids); dn,_=self.projection(nxt,ids)
        cosine=np.sum(self.heading[:,None,:]*self.unit[ids],axis=2)
        repeat=lambda x:np.broadcast_to(x[:,None],(n,k))
        rank=np.broadcast_to(np.arange(k)[None,:],(n,k))
        X=np.stack([d,rank,f,cosine,self.length[ids],repeat(self.speed),repeat(self.dt),
                    repeat(self.displacement),dp,dn,self.indeg[self.trip.arcs[ids,0]],
                    self.outdeg[self.trip.arcs[ids,1]],repeat(self.heading_ok),
                    d/(1+d[:,:1]),cosine*repeat(self.speed)],axis=2)
        return np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6).astype(np.float32)

def align_reference(g, route):
    """Chronological weak labels. This routine must NEVER be called by inference."""
    c=g.candidates(k=12,arc_pool=route)
    pos=c['pool_position']; ids=c['ids']; dist=c['distance']; frac=c['fraction']
    cumulative=np.r_[0.,np.cumsum(g.length[route])]
    progress=cumulative[pos]+frac*g.length[ids]
    emission=(dist/20.)**2
    emission+=2*np.maximum(0,-np.sum(g.heading[:,None,:]*g.unit[ids],axis=2))*g.heading_ok[:,None]
    emission[~c['valid']]=1e12
    T,K=ids.shape; back=np.zeros((T,K),np.int16); score=emission[0].copy()
    resets=np.zeros(T,bool)
    for t in range(1,T):
        delta=progress[t][None,:]-progress[t-1][:,None]
        cost=np.abs(delta-g.displacement[t])/max(20.,g.displacement[t]+10)
        cost+=np.where(delta < -10,1e6+np.abs(delta),0)
        cost+=np.maximum(0,delta-(80*g.dt[t]+150))/20
        trans=score[:,None]+cost
        best=np.argmin(trans,axis=0)
        s=trans[best,np.arange(K)]+emission[t]
        if np.min(s)>1e5 or g.dt[t]>60:
            resets[t]=True; s=emission[t].copy(); best=np.full(K,np.argmin(score))
        back[t]=best; score=s-np.min(s)
    chosen=np.zeros(T,int); chosen[-1]=np.argmin(score)
    for t in range(T-1,0,-1): chosen[t-1]=back[t,chosen[t]]
    rows=np.arange(T); labels=ids[rows,chosen]; distance=dist[rows,chosen]
    heading=np.sum(g.heading*g.unit[labels],axis=1)
    confident=(distance<=30)&(~g.heading_ok|(heading>=0.2))&(~resets)
    # Exclude ambiguous nearby reference alternatives with different physical arc IDs.
    alternatives=(ids!=labels[:,None])&c['valid']
    other=np.min(np.where(alternatives,dist,np.inf),axis=1)
    # Direction evidence may resolve bidirectional coincident roads.
    other_heading=np.sum(g.heading[:,None,:]*g.unit[ids],axis=2)
    ambiguous=np.any(alternatives&(dist<distance[:,None]+2)&(other_heading>heading[:,None]-0.3),axis=1)
    confident &= ~ambiguous
    return {'labels':labels,'confident':confident,'distance':distance,'resets':resets,
            'progress':progress[rows,chosen],'ambiguous':ambiguous}

class NetworkDistances:
    def __init__(self,g,limit=5000):
        self.g=g; self.limit=limit
        self.distance=lru_cache(maxsize=384)(self._distance)
    def _distance(self,source):
        return dijkstra(self.g.graph,directed=True,indices=source,limit=self.limit)
    def matrix(self,previous,current):
        sources=self.g.trip.arcs[previous,1]; targets=self.g.trip.arcs[current,0]
        return np.vstack([self.distance(int(s))[targets] for s in sources])

def transition(g,network,c,t,beta=35.):
    prev=c['ids'][t-1]; cur=c['ids'][t]
    d=network.matrix(prev,cur)
    travelled=(1-c['fraction'][t-1,:,None])*g.length[prev,None]+d+c['fraction'][t,None,:]*g.length[cur][None,:]
    same=prev[:,None]==cur[None,:]
    same_distance=(c['fraction'][t,None,:]-c['fraction'][t-1,:,None])*g.length[prev,None]
    travelled=np.where(same&(same_distance>=-12),np.maximum(0,same_distance),travelled)
    feasible=np.isfinite(travelled)&(travelled<=80*g.dt[t]+150)
    cost=np.abs(travelled-g.displacement[t])/max(beta,0.5*g.displacement[t]+10)
    log_transition=np.where(feasible,-cost,-np.inf)
    normalizer=logsumexp(log_transition,axis=1,keepdims=True)
    log_transition=log_transition-np.where(np.isfinite(normalizer),normalizer,0)
    return log_transition,feasible

def viterbi(g,c,probabilities,beta=35.,gap_reset=30.):
    logp=np.log(np.clip(probabilities,1e-12,1))
    logp=np.where(c['valid'],logp,-np.inf)
    T,K=logp.shape; back=np.zeros((T,K),np.int16)
    score=logp[0].copy(); resets=[]; network=NetworkDistances(g)
    for t in range(1,T):
        tr,_=transition(g,network,c,t,beta)
        joint=score[:,None]+tr
        best=np.argmax(joint,axis=0)
        s=joint[best,np.arange(K)]+logp[t]
        if not np.isfinite(s).any() or g.dt[t]>gap_reset:
            resets.append(t); best=np.full(K,np.argmax(score)); s=logp[t].copy()
        back[t]=best; score=s-np.max(s)
    state=np.zeros(T,int); state[-1]=np.argmax(score)
    for t in range(T-1,0,-1): state[t-1]=back[t,state[t]]
    return c['ids'][np.arange(T),state],state,resets

def candidate_probabilities(model,X,valid):
    p=model.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(X.shape[:2])
    p=np.where(valid,np.clip(p,1e-8,1-1e-8),0)
    return p/np.maximum(p.sum(axis=1,keepdims=True),1e-12)

def direct_continuity(g,sequence):
    if len(sequence)<2:return 1.
    a=sequence[:-1]; b=sequence[1:]
    return float(np.mean((a==b)|(g.trip.arcs[a,1]==g.trip.arcs[b,0])))

def stitch_route(g,sequence,max_connector=3000.):
    """Insert directed shortest-path connectors for display; preserve resets as breaks."""
    route=[int(sequence[0])]; breaks=[]; lookup={tuple(e):i for i,e in enumerate(g.trip.arcs)}
    cache={}
    for t,(a,b) in enumerate(zip(sequence,sequence[1:]),1):
        a=int(a);b=int(b)
        if g.dt[t]>30:
            breaks.append(len(route));route.append(b);continue
        if a==b:continue
        u=int(g.trip.arcs[a,1]);v=int(g.trip.arcs[b,0])
        if u!=v:
            key=(u,v)
            if key not in cache:
                distances,pred=dijkstra(g.graph,directed=True,indices=u,return_predecessors=True,limit=max_connector)
                connection=[]
                if np.isfinite(distances[v]):
                    q=v
                    while q!=u:
                        p=int(pred[q])
                        if p<0:break
                        connection.append(lookup[(p,q)]);q=p
                    connection.reverse()
                else:connection=None
                cache[key]=connection
            if cache[key] is None:breaks.append(len(route))
            else:route.extend(cache[key])
        route.append(b)
    return route,breaks
