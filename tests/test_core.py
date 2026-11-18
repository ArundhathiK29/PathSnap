import itertools,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from core import *

def tiny_trip():
    nodes=np.array([[37.,55.],[37.001,55.],[37.002,55.],[37.001,55.001]])
    arcs=np.array([[0,1],[1,2],[1,3],[1,0],[2,1],[3,1]])
    gps=np.array([[37.0003,55.,1],[37.0009,55.,2],[37.0016,55.,3.]])
    return Trip('00000001','train',gps,nodes,arcs)

def test_projection_in_metres_and_candidate_mapping():
    g=Geometry(tiny_trip());assert 60<g.length[0]<70
    c=g.candidates(k=4);X=g.features(c)
    assert X.shape==(3,4,len(FEATURES));assert np.isfinite(X).all()
    assert c['distance'][0,0]<1e-4
    assert all(0<=a<len(g.trip.arcs) for a in c['ids'].ravel())

def test_viterbi_matches_exhaustive_network_path():
    g=Geometry(tiny_trip());c=g.candidates(k=3)
    p=np.array([[.8,.15,.05],[.5,.3,.2],[.6,.3,.1]])
    pred,state,resets=viterbi(g,c,p)
    network=NetworkDistances(g);best=None;score=-np.inf
    for sequence in itertools.product(range(3),repeat=3):
        value=sum(np.log(p[t,s]) for t,s in enumerate(sequence))
        for t in range(1,3):value+=transition(g,network,c,t)[0][sequence[t-1],sequence[t]]
        if value>score:score=value;best=sequence
    assert not resets;assert tuple(state)==best
    assert np.array_equal(pred,c['ids'][np.arange(3),state])

def test_reference_occurrences_not_row_labels():
    g=Geometry(tiny_trip());a=align_reference(g,np.array([0,1]))
    assert len(a['labels'])==3;assert set(a['labels']).issubset({0,1})

def test_invalid_trip_id_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):load_trip(tmp_path,'train','../bad')

def test_duplicate_edges_do_not_double_network_distance():
    t=tiny_trip();t.arcs=np.vstack([t.arcs,t.arcs[:1]])
    g=Geometry(t);assert np.isclose(g.graph[0,1],g.length[0])

def test_transition_rows_are_probabilities():
    g=Geometry(tiny_trip());c=g.candidates(k=3)
    tr,valid=transition(g,NetworkDistances(g),c,1)
    rows=valid.any(1);assert np.allclose(np.exp(tr[rows]).sum(1),1)

def test_inference_source_excludes_references():
    import ast
    tree=ast.parse((Path(__file__).resolve().parents[1]/'api.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
    calls={n.func.id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not {'load_reference','align_reference'}&calls
