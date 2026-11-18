# Data dictionary and metric definitions

Raw files: `.track` longitude/latitude/time seconds; `.nodes` longitude/latitude with zero-based node index; `.arcs` directed endpoint node IDs with zero-based arc index; `.route` ordered arc occurrences. All whitespace delimited and headerless. Parser rejects nonfinite values, nonintegral or out-of-range graph IDs, and nonincreasing timestamps. No rows silently dropped.

Derived labels: nearest plausible ordered reference occurrences aligned with dynamic programming on cumulative reference length. Distance emission and heading agreement; backward progress penalised; progress jumps penalised relative to elapsed time and observed displacement; large gaps restart alignment. Confident subset requires <=30m residual, compatible heading when displacement supports a heading, no reset, and no unresolved near-equal competing reference arc. Repeated arc occurrences retain separate positions in alignment. They map back to the original arc ID afterwards. This creates weak labels, not new independent truth. Report scored coverage and full-sequence agreement separately. Ambiguous and unconfident points still receive model predictions but are excluded from the primary pointwise score.

Feature names and units (ordered exactly as in the model):

| Feature | Meaning |
|---|---|
| distance_m | GPS point to candidate line-segment distance, metres |
| distance_rank | Candidate rank after distance reranking, zero based |
| projection_fraction | Clamped projection along directed segment, [0,1] |
| heading_cosine | Dot product of local GPS direction and segment direction |
| segment_length_m | Directed segment length in local metre projection |
| gps_speed_mps | Previous GPS displacement / time delta, capped at 100 m/s |
| time_gap_s | Previous time delta; first observation uses 1s |
| displacement_m | Previous GPS displacement; first observation uses zero |
| previous_distance_m | Previous GPS point to this candidate segment |
| next_distance_m | Next GPS point to this candidate segment; last repeats current |
| in_degree | Incoming raw-arc count at candidate start node |
| out_degree | Outgoing raw-arc count at candidate end node |
| heading_available | Local centred displacement >4m |
| distance_over_best | Candidate distance / (1m + closest candidate distance) |
| speed_heading_interaction | Heading cosine multiplied by GPS speed |

Whole-trip offline reconstruction uses next GPS observations; this is not an online streaming claim. No reference-derived features, arc IDs, trip IDs or absolute coordinates enter the classifier.

Candidate binary Macro F1: unweighted average of positive-match and negative-match F1. Candidate precision/recall use threshold 0.5 on binary compatibility probability. Candidate log loss uses raw model probabilities. Top-1 reference-derived agreement: fraction of confidently aligned observations whose highest-scored candidate matches the weak reference arc ID. Candidate misses count as errors in Top-1, not silently excluded. Candidate Miss Rate: reference arc absent from valid candidate set among confident labels. Aggregated Top-1 is observation-weighted; continuity and candidate-miss summaries in model table are macro averages over trips.

Direct continuity: proportion of consecutive observation predictions that stay on the same arc or have exactly adjoining directed endpoints. This deliberately strict metric can penalise legitimate movement over intervening arcs. Stitched arc-set precision/recall compare traversed arc sets with supplied reference arc sets; they ignore order and must not be called ordered sequence accuracy. Decoder results also preserve per-observation sequence and explicit route breaks. No single metric is presented as complete route quality.

Geometry limitation: local equirectangular city-scale approximation, not a full ellipsoidal geodesic model. GPS heading is noisy during stops. Duplicate directed arcs with identical endpoints are observationally indistinguishable; exact arc-ID agreement can therefore penalise geometrically equivalent choices. Reference discontinuities and large GPS gaps limit certainty. Candidate retrieval is approximate spatial retrieval plus exact reranking, not an exhaustive nearest-segment guarantee.
