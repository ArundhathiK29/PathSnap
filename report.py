"""Build the consolidated technical report exclusively from recorded experiments."""
import json,html
from pathlib import Path
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader

HERE=Path(__file__).parent;R=HERE/'results'
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='SmallText',fontSize=8,leading=11,spaceAfter=5))
styles['BodyText'].fontSize=9;styles['BodyText'].leading=13;styles['BodyText'].spaceAfter=8
styles['Heading1'].textColor=colors.HexColor('#123e52');styles['Heading1'].spaceBefore=14
styles['Heading2'].textColor=colors.HexColor('#176a78')
story=[]
def p(text,style='BodyText'):story.append(Paragraph(text,styles[style]))
def heading(text):p(text,'Heading1')
def table(rows,widths=None):
    data=[[Paragraph(html.escape(str(c)),styles['SmallText']) for c in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#6b8797')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')])]))
    story.extend([t,Spacer(1,10)])
def figure(name,caption):
    path=R/'figures'/name
    if path.exists():
        w,h=ImageReader(str(path)).getSize();width=480;height=min(350,width*h/w)
        story.append(Image(str(path),width=height*w/h,height=height));p(caption,'SmallText')
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'));canvas.line(42,35,553,35)
    canvas.setFont('Helvetica',8);canvas.drawString(42,23,'PathSnap | Evidence-based technical report | PES University UE24CS352A');canvas.drawRightString(553,23,str(doc.page))

def main():
    inventory=pd.read_csv(R/'inventory.csv',dtype={'trip_id':str});metrics=pd.read_csv(R/'model_metrics.csv');summary=json.loads((R/'data_summary.json').read_text());tuning=json.loads((R/'tuning_summary.json').read_text());decode=pd.read_csv(R/'decoder_metrics.csv',dtype={'trip_id':str});selection=json.loads((R/'selection.json').read_text());env=json.loads((R/'environment.json').read_text())
    p('PATHSNAP','Title');p('Distributed map matching and route reconstruction','Heading2')
    p('PES University - UE24CS352A Machine Learning Hackathon')
    p('Consolidated technical report generated from actual saved experiments. AI-assisted implementation; team review and reproduction required before submission.')
    verdict=json.loads((R/'readiness.json').read_text()) if (R/'readiness.json').exists() else {'verdict':'NOT READY','reason':'Final audit incomplete'}
    p('Readiness: <b>'+verdict['verdict']+'</b>. '+html.escape(verdict.get('reason','')))
    p('<b>Metric limitation:</b> supplied reference routes are sequences of traversed arcs, not observation-level labels. All pointwise scores below are agreement against explicitly constructed reference-derived weak labels. They must not be described as independently verified GPS-point accuracy. Final organiser scoring and the ten reserved trips remain unverified.')
    heading('1. Problem and architecture')
    p('Noisy GPS observations are matched to a directed road network. Spatial candidate retrieval and 15 features feed a binary candidate classifier; its normalised scores become emissions for an HMM. Network-distance transitions and log-domain Viterbi recover a feasible sequence. Directed shortest paths supply intermediate road segments where feasible. A FastAPI server performs inference and separate post-prediction evaluation; a second-laptop UI selects trips and renders returned geometry.')
    table([['Component','Responsibility'],['Laptop 1','GPS/graph, trained artifacts, candidate features, HMM/Viterbi, API, private reference scorer'],['Laptop 2','Trip selector, real API requests, predicted geometry, aggregate metrics and error states']],[95,385])
    story.append(PageBreak())
    heading('2. Dataset understanding and exploratory analysis')
    rows=[['Partition','Trips','GPS points','Confident labels','Gaps >1s','Reference breaks']]
    for part,df in inventory.groupby('partition'):
        rows.append([part,len(df),int(df.gps_count.sum()),int(df.confident_labels.sum()),int(df.gaps_gt_1s.sum()),int(df.reference_discontinuities.sum())])
    table(rows,[85,40,80,90,75,100])
    p('Dataset consists of 360 headerless whitespace-delimited files. Tracks contain longitude, latitude and time; nodes contain longitude and latitude; arcs contain directed node IDs; routes contain ordered arc IDs. Indices are zero-based. Numeric shapes, finite values, coordinate plausibility, endpoint bounds and strictly increasing timestamps are validated. No observations were silently removed. Missing GPS values: '+str(int(inventory.gps_missing.sum()))+'.')
    p('Duplicate coordinate observations are retained because a stationary vehicle can generate repeated coordinates. Irregular gaps use their actual time delta. Weak/strong connected-component counts, geographical extents, zero-length arcs, repeated positions and label coverage are retained per trip in inventory.csv. Duplicate directed edges are collapsed only for shortest-path distances; original arc IDs remain intact. Otherwise SciPy CSR would sum duplicates and inflate distances.')
    p('A local spherical equirectangular projection, centred on each trip network, converts coordinates to metres. Longitude is scaled by cos(latitude). This is a documented local approximation suitable for city-scale feature distances; it is not an ellipsoidal geodesic claim. Folium receives latitude/longitude ordering; raw files use longitude/latitude.')
    figure('eda_summary.png','Figure 1. Actual trip-level GPS counts, maximum timestamp gaps and confidence coverage. Both development partitions are identified in the legend.')
    figure('eda_example.png','Figure 2. Training trip 00000001: GPS observations, road graph and supplied reference route, in local metre coordinates.')
    heading('2.1 Reference alignment and uncertainty')
    p('Reference arc occurrences are kept in chronological order, with cumulative route lengths. For each GPS point, nearby reference occurrences are proposed. Dynamic programming combines squared point-to-segment residual, heading agreement and progress costs, penalising backward progress and implausible jumps. Large gaps or impossible progress restart the alignment. This routine is isolated from inference.')
    p('A label is counted as confident only if residual distance is at most 30m, heading is compatible when informative, the point is not a reset, and nearby competing reference arcs are not unresolved. Low-confidence points still receive predictions but are excluded from primary pointwise scoring. Coverage varies substantially by trip; the full inventory exposes that selection effect. Reported scores can therefore be optimistic relative to all observations. Identical duplicate directed arcs are geometrically indistinguishable but may have different IDs, limiting exact-ID scoring. Reference routes are not assumed error-free.')
    heading('2.2 Candidate generation and features')
    p('All road segments are sampled at no more than 25m spacing. A spatial tree retrieves nearby samples, with midpoint retrieval as fallback. Retrieved unique arcs are reranked using exact clamped point-to-segment distance; eight candidate slots are used. This is approximate retrieval followed by exact reranking, not a proof of exhaustive nearest-segment recall.')
    p('Features: candidate distance and rank, projection fraction, heading cosine, segment length, GPS speed, time gap, displacement, previous/next GPS distance to candidate, input-graph in/out degree, heading availability, distance relative to nearest candidate, and speed-heading interaction. No arc IDs, trip IDs, absolute coordinates or reference data are classifier inputs. Future GPS is available because the task reconstructs a complete recorded trip; no real-time streaming claim is made.')
    p('Training uses up to 240 eligible observations from each of the 80 training trips, seeded at 2911. All candidates for each sampled observation are retained. Actual training candidate rows: '+str(summary['training_candidate_rows'])+'. This bounded sampling is a computational tradeoff, not full-data training. Student-test evaluation runs on every observation of all ten trips, with confident-label coverage reported.')
    story.append(PageBreak())
    heading('3. Model experimentation and comparison')
    p('Three-fold GroupKFold holds out whole trips; every training trip appears in one validation fold. Learned scaling is fitted inside each fold. Candidate-level binary Macro F1 is the selection objective. Top-1 agreement chooses the highest-scoring candidate per observation; candidate misses are counted as errors. Threshold-based candidate F1 and precision/recall use 0.5 binary probability. These are different tasks and are reported separately.')
    rows=[['Model','CV Macro F1','Test Macro F1','Top-1 agreement','Candidate log loss','Direct continuity','Predict seconds']]
    for _,r in metrics.iterrows():rows.append([r.model,f'{r.cv_macro_f1:.4f}',f'{r.student_macro_f1:.4f}',f'{r.top1_reference_agreement:.4f}',f'{r.candidate_binary_log_loss:.4f}',f'{r.direct_route_continuity:.4f}',f'{r.inference_seconds:.3f}'])
    table(rows,[105,60,62,66,65,65,57])
    p('Test scores above use '+str(int(metrics.iloc[0].scored_observations))+' confident observations out of 19,160 student-test observations. Candidate miss rate (macro over trips): '+f'{metrics.iloc[0].candidate_miss_rate:.4%}'+'. Prediction timing covers classifier scoring, not candidate preparation, decoding or network transport. Direct continuity and candidate misses are macro averages across trips; Top-1 is observation-weighted.')
    figure('model_comparison.png','Figure 3. Actual model comparisons. No synthetic learning curves are shown for algorithms without epoch histories.')
    descriptions={
      'LogisticRegression':('Linear/probabilistic','Logistic log loss with L2 regularisation','Fast interpretable baseline on distance, heading and motion evidence. Limited nonlinear interactions; feature scaling is necessary.'),
      'GaussianNB':('Probabilistic','Gaussian class-conditional likelihood with conditional-independence assumption','Very fast baseline. Correlated geometric features violate conditional independence; high positive recall can come with poor precision and overconfident probabilities.'),
      'DecisionTree':('Tree','Greedy Gini impurity reduction','Captures thresholds and interactions without scaling. A single tree can overfit and gives coarse or extreme leaf probabilities.'),
      'RandomForest':('Ensemble','Bootstrap trees with Gini splits and probability averaging','Reduces single-tree variance and handles nonlinear features. Slower fitting and larger artifacts; depth/leaf controls matter.'),
      'HistGradientBoosting':('Boosting','Binary log loss, additive histogram-based boosting','Efficient nonlinear classifier. Learning rate, leaf count and regularisation control fit; probabilities still require empirical diagnosis.')}
    for _,r in metrics.iterrows():
        name=r.model;family,loss,why=descriptions[name];p(name,'Heading2')
        cfg=json.loads((R/f'{name}_configuration.json').read_text());display={k:v for k,v in cfg.items() if k in ['C','max_iter','max_depth','min_samples_leaf','n_estimators','max_features','max_leaf_nodes','learning_rate','l2_regularization','gaussiannb__var_smoothing','logisticregression__C','logisticregression__max_iter']}
        p(f'<b>Family:</b> {family}. <b>Objective:</b> {loss}. {why}')
        p('Important initial settings: '+html.escape(json.dumps(display))+'. Full configuration is retained in results/.')
        gap=r.train_macro_f1-r.cv_macro_f1
        p(f'Actual train Macro F1 {r.train_macro_f1:.4f}; grouped CV {r.cv_macro_f1:.4f}; student-test Macro F1 {r.student_macro_f1:.4f}; Top-1 agreement {r.top1_reference_agreement:.4f}; precision {r.precision:.4f}; recall {r.recall:.4f}; candidate log loss {r.candidate_binary_log_loss:.4f}; final-fit time {r.training_seconds:.2f}s. Train-CV gap {gap:.4f}. '+('The train-CV gap indicates some fitting advantage on seen trips; held-out trip results provide the generalisation check.' if gap>.008 else 'The modest train-CV gap gives no large overfitting signal under this sampling and weak-label protocol. It does not establish absence of bias.'))
    p('Selected model: <b>'+selection['model']+'</b>, selected using training-trip CV rather than training accuracy or reserved data. Probability quality, runtime and continuity are retained as diagnostics. No external calibration set or calibrated-probability claim is made.')
    story.append(PageBreak())
    heading('4. Bayesian hyperparameter optimisation')
    best=tuning['best'];p('A Gaussian-process surrogate with Matérn kernel models mean grouped CV Macro F1. Three initial seeded random designs are followed by expected-improvement acquisition over 256 proposed unit-cube points per trial. This is a genuine sequential Bayesian search, implemented using scikit-learn rather than relabelling grid search. All actual trial points, parameters, fold membership and scores are saved.')
    p('Histogram boosting search: leaf nodes 7-47, learning rate 10^[-1.7,-0.6], L2 regularisation 10^[-2,2]. Other algorithm-specific mappings are defined in workflow.decode_parameters. Budget: '+str(tuning['trials'])+' completed trials, three trip-grouped folds per trial. Stopping rule: fixed trial budget. Seed 2912 for search proposals. No evaluator data is used.')
    p('Best parameters: '+html.escape(json.dumps(best['parameters']))+'. Search computational cost: '+f"{tuning['total_seconds']:.2f}"+'s including final fit and student scoring.')
    base=metrics[metrics.model==tuning['model']].iloc[0];tm=tuning['student_metrics']
    table([['Metric','Untuned','Tuned'],['CV Macro F1',f'{base.cv_macro_f1:.5f}',f"{best['cv_macro_f1']:.5f}"],['Student Top-1 agreement',f'{base.top1_reference_agreement:.5f}',f"{tm['top1_reference_agreement']:.5f}"],['Student Macro F1',f'{base.student_macro_f1:.5f}',f"{tm['student_macro_f1']:.5f}"],['Candidate log loss',f'{base.candidate_binary_log_loss:.5f}',f"{tm['candidate_binary_log_loss']:.5f}"],['Classifier inference seconds',f'{base.inference_seconds:.3f}',f"{tm['inference_seconds']:.3f}"]],[220,130,130])
    p('Tuning '+('improved' if best['cv_macro_f1']>base.cv_macro_f1 else 'did not improve')+' the selection objective. Student-test behaviour is reported independently; CV improvement does not guarantee improvement in every metric.')
    heading('5. HMM and Viterbi reconstruction')
    p('Hidden states are the candidate arc IDs at each observation. Emissions are nonnegative classifier compatibility probabilities normalised over that observation\'s valid candidates. Initial states have a uniform prior (a constant term in decoding). Transition probabilities use exp(-absolute(network travel distance - observed GPS displacement)/scale), row-normalised over feasible next candidates. Network travel combines remaining previous-segment length, directed shortest-path distance and progress along the next segment; same-segment progress is treated separately.')
    p('Scale is max(35m, half observed displacement + 10m). Implausible transitions above 80m/s times elapsed time +150m tolerance are forbidden. Bounded directed shortest paths use a 5km search limit. Viterbi uses log probabilities, max recurrence, score recentering and stored backpointers. Missing candidate slots are masked; nearest-segment fallback normally supplies candidates. Entirely invalid transition steps and gaps over 30s restart the chain, with reset counts retained. Shortest-path stitching inserts intervening arcs for geometry, and leaves explicit breaks for long gaps or unreachable connectors.')
    rows=[['Trip','Label coverage','Classifier agreement','Decoded agreement','Raw continuity','Decoded continuity','Resets']]
    for _,r in decode.iterrows():
        gps=int(inventory[(inventory.partition=='student_test')&(inventory.trip_id==r.trip_id)].iloc[0].gps_count)
        rows.append([r.trip_id,f'{r.scored_observations/gps:.1%}',f'{r.classifier_top1_agreement:.4f}',f'{r.decoder_top1_agreement:.4f}',f'{r.classifier_direct_continuity:.4f}',f'{r.decoder_direct_continuity:.4f}',int(r.decoder_resets)])
    table(rows,[70,65,80,80,65,70,50])
    raw=decode.classifier_correct.sum()/decode.scored_observations.sum();dec=decode.decoder_correct.sum()/decode.scored_observations.sum()
    p(f'Observation-weighted comparison: classifier {raw:.4%}; classifier + HMM/Viterbi {dec:.4%}. Improved trips: {int((decode.decoder_top1_agreement>decode.classifier_top1_agreement).sum())}; degraded trips: {int((decode.decoder_top1_agreement<decode.classifier_top1_agreement).sum())}. The decoder is not claimed to improve every trip. Transition assumptions, ambiguous labels, gaps and duplicate arcs can reduce agreement even when topology becomes more coherent.')
    p('Per-trip stitched arc-set precision/recall and route-break counts appear in decoder_metrics.csv. Arc-set metrics ignore order; direct continuity requires identical or adjoining arcs and can penalise valid skipped intermediate segments. Neither is misrepresented as complete route-sequence accuracy.')
    if 'ordered_arc_matching_ratio' in decode:
        p(f"Mean order-sensitive arc matching-block ratio (difflib.SequenceMatcher, autojunk disabled): {decode.ordered_arc_matching_ratio.mean():.4f}. This compares the stitched arc sequence with the chronological reference, including repeated occurrences. It is a matching-block similarity, not optimal edit distance or exact-route accuracy.")
    figure('prediction_example.png','Figure 4. Actual student-test trip 00000000: GPS, supplied reference, and decoded prediction. This is a Matplotlib geographical diagnostic, not a fabricated Folium screenshot.')
    heading('6. Folium geographical visualisation')
    backend=json.loads((R/'map_backend.json').read_text());p('Actual map backend: '+backend['kind']+'. Interactive HTML maps in results/maps show GPS observations, local road geometry, reference route for authorised analysis and predicted segments. Layer colours and legend distinguish them, with controls and bounds fitted to the actual data. Upstream Folium/Branca/xyzservices sources and licences are included because package installation was unavailable. No third-party mapping implementation is claimed as original team code.')
    screenshot=R/'figures'/'folium_screenshot.png'
    if screenshot.exists():figure('folium_screenshot.png','Figure 5. Actual captured Folium map rendering.')
    else:p('<b>Outstanding evidence:</b> a captured Folium browser screenshot/export has not been included. Generated HTML and the static geographical figure are available, but the required screenshot should be captured during team review.')
    heading('7. API, UI and deployment')
    p('FastAPI exposes /health, /trips and POST /predict with an eight-digit trip identifier. Dataset partition is selected by server mode; requests cannot supply filesystem paths. Inference loads only GPS, graph and trusted saved model artifacts. Reference loading happens afterwards in a separate scorer. Only aggregate scores and predicted geometry reach the UI. Logs identify received trip requests. A busy lock prevents concurrent memory-heavy decoding. Exact CORS origins are configurable; the service is intended for a trusted local demonstration network.')
    for image,caption in [('ui_screenshot.png','Actual UI displaying a real API response.'),('api_screenshot.png','Actual API/log evidence captured during local verification.')]:
        if (R/'figures'/image).exists():figure(image,caption)
    p('Physical two-laptop deployment has not been verified. Local API/UI checks cannot establish cross-laptop reachability. Reserved evaluator trips have not been provided; evaluator mode returns zero trips rather than pretending development trips are reserved tests. README documents addresses, CORS configuration, private-network binding, startup commands and expected request logs. Publication of a public GitHub repository remains outstanding unless separately verified in FINAL_AUDIT.md.')
    heading('8. Reproducibility, provenance and final verification')
    p('Environment: '+html.escape(json.dumps(env))+'. Full configurations, fold manifests, Bayesian history, saved artifacts and per-trip predictions are included. Tests compare Viterbi with exhaustive search on a tiny graph and check duplicate-edge distances, transition normalisation and inference/reference isolation. verify.py reloads the real model, reproduces metrics, refits its configuration and exercises the actual validated API handler for all ten accessible student-test trips. HTTP transport and browser integration remain unverified: Windows sandbox socket restrictions also prevented the asynchronous TestClient event loop from starting.')
    verification=json.loads((R/'verification.json').read_text()) if (R/'verification.json').exists() else []
    if verification:table([['Verification status','Count']]+[[s,sum(x['status']==s for x in verification)] for s in ['PASS','FAIL','UNVERIFIED']],[240,240])
    p('Every final audit item and its evidence is recorded in FINAL_AUDIT.md. External plagiarism-checker verification has not been performed. No similarity percentage is invented. No other team\'s code, weights, configurations or results were used. Team members must be able to explain and reproduce the submitted implementation and must disclose AI assistance according to policy.')
    p('Sources: official PES PathSnap Main Guidelines; Dataset Exploration; Models and Training Guidelines; Deliverables; Assessment and Plagiarism Policy. Standard library references: scikit-learn.org/stable/modules/gaussian_process.html; python-visualization.github.io/folium/v0.20.0/getting_started.html. Upstream source commits and licences: vendor/MANIFEST.json. Reference video inaccessible; its content is not assumed.')
    doc=SimpleDocTemplate(str(HERE/'report.pdf'),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=47,title='PathSnap technical report',author='PathSnap project - AI-assisted team review required')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(HERE/'report.pdf')

if __name__=='__main__':main()
