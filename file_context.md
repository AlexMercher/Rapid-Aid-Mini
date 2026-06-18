# RapidAid — File Context Map
> Last updated: 2026-06-18 (Phase 7 — Florence-2-large replaces bakllava as semantic narration engine)
> Line numbers approximate — always grep to confirm before editing.

---

## RapidAid Physical Engine

### `RapidAid-Accident-Detection-System/pipeline/track_processor.py`
Primary responsibility: Track-centric video processing — YOLO detection + ByteTrack + multi-signal fusion + causal state machine
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `is_portrait_frame()` | ~46 | `TrackCentricProcessor._compute_geometry_score()` | Returns True if frame is portrait orientation (h > w) |
| `portrait_swap_bboxes()` | ~54 | `TrackCentricProcessor._compute_geometry_score()` | Swaps x/y axis in bboxes for portrait geometry IoU computation |
| `TrackCentricProcessor.__init__()` | ~86 | `pipeline_manager.py:_get_rapidaid()` | Loads YOLO models (seg+pose), initializes trackers, analyzers, fusion, causal gate |
| `TrackCentricProcessor.process_video()` | ~170 | `pipeline_manager.py:process_video()` | Main loop: reads video, runs detection/tracking/analysis/fusion per frame, returns result dict |
| `TrackCentricProcessor._detect_vehicles()` | ~602 | `process_video()` | Runs YOLO seg model, filters by vehicle class/confidence/area |
| `TrackCentricProcessor._detect_persons()` | ~638 | `process_video()` | Runs YOLO pose model, returns person detections |
| `TrackCentricProcessor._compute_detector_score()` | ~665 | `process_video()` | Blends YOLO detection confidence with M1/M4 scores |
| `TrackCentricProcessor._compute_velocity_score()` | ~678 | `process_video()` | Returns max velocity collapse or trajectory anomaly score |
| `TrackCentricProcessor._compute_disappearance_score()` | ~687 | `process_video()` | Returns max disappearance score, filtering out occlusion deaths |
| `TrackCentricProcessor._has_velocity_collapse()` | ~728 | `_is_occluded_death()` | Static method: returns True if track speed dropped >70% |
| `TrackCentricProcessor._is_occluded_death()` | ~752 | `_compute_disappeared_score()` | Static method: determines if death was occlusion (passive) or genuine event |
| `TrackCentricProcessor._compute_geometry_score()` | ~792 | `process_video()` | Computes geometry from M4 zones + pairwise IoU, handles portrait swap |
| `TrackCentricProcessor._compute_iou()` | ~831 | `_compute_geometry_score()` | Static method: IoU between two bboxes |
| `TrackCentricProcessor._draw_overlay()` | ~846 | `process_video()` | Draws debug overlay with tracks, signals, banner |
| `TrackCentricProcessor._extract_state_transitions()` | ~572 | `process_video()` | Extracts CLEAR→SUSPICIOUS and SUSPICIOUS→CONFIRMED transitions |

### `RapidAid-Accident-Detection-System/pipeline/video_processor.py`
Primary responsibility: Frame-centric hybrid video scanning with classifier-based alert triggering
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `VideoProcessor.__init__()` | ~50 | `main.py:process_video()` | Initializes FrameProcessor and TemporalClassifier |
| `VideoProcessor.process_video()` | ~57 | `main.py:process_video()` | Main loop: classifier-based alert mode + full pipeline + temporal confirmation |
| `VideoProcessor._compute_motion_score()` | ~419 | `process_video()` | Mean absolute pixel difference between consecutive frames |
| `VideoProcessor._is_motion_spike()` | ~431 | `process_video()` | Detects significant motion spike (absolute + relative threshold) |

### `RapidAid-Accident-Detection-System/pipeline/frame_processor.py`
Primary responsibility: Single-frame processing through full pipeline (M1→M2→M3→M4→victims→zone→report)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `FrameProcessor.__init__()` | ~55 | `video_processor.py`, `main.py` | Loads all detection modules (vehicle, person, accident, victim, frame, damage, collision, zone, report) |
| `FrameProcessor.process()` | ~82 | `video_processor.py:process_video()` | Full single-frame pipeline: detect→classify→score→zone→report |
| `FrameProcessor._refine_involved_by_damage()` | ~487 | `process()` | Uses M2 damage scores to refine which vehicles are involved |
| `FrameProcessor._filter_weak_involved()` | ~592 | `process()` | Removes low-confidence/outlier involved vehicles |
| `FrameProcessor._empty_result()` | ~651 | `process()` | Returns empty no-accident result |

### `RapidAid-Accident-Detection-System/pipeline/report_generator.py`
Primary responsibility: Builds JSON reports and annotated frames from analysis results
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `ReportGenerator.generate()` | ~15 | `frame_processor.py:process()` | Generates report dict + annotated frame from analysis results |
| `ReportGenerator.save()` | ~69 | `video_processor.py:process_video()` | Saves annotated frame and JSON report to disk |
| `ReportGenerator._build_report()` | ~95 | `generate()` | Builds structured JSON report dict |

### `RapidAid-Accident-Detection-System/models/confidence_fusion.py`
Primary responsibility: Multi-signal weighted confidence fusion (detector + tracking + velocity + flow + disappearance + geometry)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `ConfidenceFusion.__init__()` | ~57 | `track_processor.py:__init__()` | Initializes fusion weights, validates they sum to ~1.0 |
| `ConfidenceFusion.compute()` | ~81 | `track_processor.py:process_video()` | Computes weighted sum of all signal scores into final confidence |
| `ConfidenceFusion.compute_tracking_score()` | ~144 | `track_processor.py:process_video()` | Computes tracking consistency score from active tracks |
| `ConfidenceFusion.compute_detector_score()` | ~192 | (available for direct use) | Computes aggregate detector confidence from involved vehicles |
| `ConfidenceFusion.compute_geometry_score()` | ~218 | (available for direct use) | Computes geometry score from crash scores |
| `ConfidenceFusion.adaptive_weights()` | ~234 | (available for direct use) | Redistributes weights when some signals are unavailable |

### `RapidAid-Accident-Detection-System/models/disappearance_analyzer.py`
Primary responsibility: Classifies track disappearances as normal_exit, temp_occlusion, or anomalous
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `DisappearanceAnalyzer.__init__()` | ~48 | `track_processor.py:__init__()` | Initializes with frame dimensions and thresholds |
| `DisappearanceAnalyzer.update_frame_size()` | ~68 | `track_processor.py:process_video()` | Updates frame dimensions for new video |
| `DisappearanceAnalyzer.analyze()` | ~73 | `track_processor.py:process_video()` | Analyzes all dead/lost tracks for disappearance classification |
| `DisappearanceAnalyzer.get_anomalous_disappearances()` | ~358 | (available for direct use) | Filters anomalous disappearances above threshold |

### `RapidAid-Accident-Detection-System/models/causal_gate.py`
Primary responsibility: Causal intelligence gate — state machine requiring physical evidence before accident confirmation
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `CausalGate.__init__()` | ~37 | `track_processor.py:__init__()` | Initializes thresholds, confirm frames required, evidence families |
| `CausalGate.reset()` | ~64 | `track_processor.py:process_video()` | Resets state machine to CLEAR |
| `CausalGate.evaluate()` | ~71 | `track_processor.py:process_video()` | Evaluates signals, advances state machine: CLEAR→SUSPICIOUS→CONFIRMED→AFTERMATH |

### `RapidAid-Accident-Detection-System/models/tracker.py`
Primary responsibility: Multi-object tracking with persistent IDs (ByteTrack-style)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `Track.__init__()` | ~30 | `TrackManager.update()` | Single object track with bbox, velocity, history, lifecycle |
| `Track.update()` | ~79 | `TrackManager.update()` | Updates track with new detection, computes velocity/acceleration |
| `Track.get_speed()` | ~100+ | `track_processor.py:process_video()` | Returns current speed magnitude |
| `TrackManager.__init__()` | ~100+ | `track_processor.py:__init__()` | Initializes tracker with max_lost_frames and IoU threshold |
| `TrackManager.update()` | ~100+ | `track_processor.py:process_video()` | Matches detections to tracks, creates new tracks, manages lifecycle |
| `TrackManager.get_all_lost_tracks()` | ~100+ | `track_processor.py:process_video()` | Returns all currently lost tracks |
| `TrackManager.get_recently_dead_tracks()` | ~100+ | `track_processor.py:process_video()` | Returns tracks that died within lookback window |
| `TrackManager.get_all_active_tracks()` | ~100+ | `track_processor.py:_draw_overlay()` | Returns all active tracks |

### `RapidAid-Accident-Detection-System/models/velocity_analyzer.py`
Primary responsibility: Detects velocity collapse, sudden stops, and trajectory anomalies
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `VelocityAnalyzer.__init__()` | ~31 | `track_processor.py:__init__()` | Initializes speed/decel/direction thresholds |
| `VelocityAnalyzer.analyze_tracks()` | ~51 | `track_processor.py:process_video()` | Analyzes all tracks for velocity and trajectory anomalies |

### `RapidAid-Accident-Detection-System/models/optical_flow_analyzer.py`
Primary responsibility: Farneback optical flow analysis for motion burst detection (YOLO-independent)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `OpticalFlowAnalyzer.__init__()` | ~34 | `track_processor.py:__init__()` | Initializes grid size, flow scale, Farneback parameters |
| `OpticalFlowAnalyzer.reset()` | ~71 | `track_processor.py:process_video()` | Resets state for new video |
| `OpticalFlowAnalyzer.process_frame()` | ~77 | `track_processor.py:process_video()` | Computes optical flow and returns motion burst + flow scores |

### `RapidAid-Accident-Detection-System/models/camera_stabilizer.py`
Primary responsibility: Detects camera shake by analyzing global flow coherence
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `CameraStabilizer.__init__()` | ~23 | `track_processor.py:__init__()` | Initializes coherence threshold, grid size |
| `CameraStabilizer.compute_suppression()` | ~36 | `track_processor.py:process_video()` | Returns suppression factor (0=full shake, 1=stable) |

### `RapidAid-Accident-Detection-System/models/collision_detector.py`
Primary responsibility: M4 — Direct collision zone detection using fine-tuned YOLOv8n
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `CollisionDetector.__init__()` | ~57 | `frame_processor.py:__init__()`, `track_processor.py:__init__()` | Loads collision detector model if available |
| `CollisionDetector.is_available()` | ~78 | `frame_processor.py:process()`, `video_processor.py:process_video()` | Checks if M4 model is loaded |
| `CollisionDetector.detect()` | ~80+ | `frame_processor.py:process()`, `track_processor.py:process_video()` | Detects collision zones in frame |
| `CollisionDetector.get_involved_vehicles()` | ~80+ | `frame_processor.py:process()` | Finds vehicles closest to collision zone centers |

### `RapidAid-Accident-Detection-System/models/frame_classifier.py`
Primary responsibility: M1 — Binary scene classifier (accident/no_accident) using YOLOv8n-cls
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `FrameClassifier.__init__()` | ~29 | `frame_processor.py:__init__()`, `video_processor.py:__init__()` | Loads accident classifier model if available |
| `FrameClassifier.is_available()` | ~59 | `frame_processor.py:process()`, `video_processor.py:process_video()` | Checks if M1 model is loaded |
| `FrameClassifier.classify()` | ~60+ | `frame_processor.py:process()`, `video_processor.py:process_video()` | Returns (is_accident, confidence) for a frame |

### `RapidAid-Accident-Detection-System/models/vehicle_detector.py`
Primary responsibility: Vehicle detection using YOLOv8s-seg with background filtering
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `VehicleDetector.__init__()` | ~24 | `frame_processor.py:__init__()` | Loads YOLO vehicle detection model |
| `VehicleDetector.detect()` | ~36 | `frame_processor.py:process()` | Detects vehicles, returns list of vehicle dicts with bbox/polygon/confidence |

### `RapidAid-Accident-Detection-System/models/person_detector.py`
Primary responsibility: Person detection using YOLOv8s-pose with keypoints
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `PersonDetector.__init__()` | ~33 | `frame_processor.py:__init__()` | Loads YOLO pose model |
| `PersonDetector.detect()` | ~45 | `frame_processor.py:process()` | Detects persons with keypoints, returns list of person dicts |

### `RapidAid-Accident-Detection-System/models/accident_classifier.py`
Primary responsibility: Multi-signal crash scoring for vehicle-vehicle collision classification
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `AccidentClassifier.__init__()` | ~32 | `frame_processor.py:__init__()` | Loads crash score weights and threshold |
| `AccidentClassifier.classify()` | ~36 | `frame_processor.py:process()` | Analyzes vehicle pairs, returns involved vehicles with crash_scores |
| `AccidentClassifier._check_single_vehicle_accident()` | ~58+ | `frame_processor.py:process()` | Checks single-vehicle crash with classifier confidence |

### `RapidAid-Accident-Detection-System/models/victim_classifier.py`
Primary responsibility: Classifies persons as victims or bystanders using keypoints + spatial analysis
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `VictimClassifier.classify()` | ~22 | `frame_processor.py:process()` | Classifies all persons, returns victim dicts with status |
| `VictimClassifier.detect_standalone_victims()` | ~40 | `frame_processor.py:process()` | Detects victims lying on ground without vehicle involvement |

### `RapidAid-Accident-Detection-System/models/damage_classifier.py`
Primary responsibility: M2 — Per-vehicle damage classification using YOLOv8n-cls
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `DamageClassifier.__init__()` | ~43 | `frame_processor.py:__init__()` | Loads damage classifier model if available |
| `DamageClassifier.is_available()` | ~60+ | `frame_processor.py:process()` | Checks if M2 model is loaded |
| `DamageClassifier.classify_all_vehicles()` | ~60+ | `frame_processor.py:process()` | Classifies all vehicles, returns bbox→damage_score mapping |

### `RapidAid-Accident-Detection-System/models/temporal_classifier.py`
Primary responsibility: M3 — LSTM-based temporal sequence classifier for accident detection
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `TemporalClassifier.__init__()` | ~60+ | `video_processor.py:__init__()` | Loads temporal classifier model if available |
| `TemporalClassifier.is_available()` | ~60+ | `video_processor.py:process_video()` | Checks if M3 model is loaded |
| `TemporalClassifier.reset()` | ~60+ | `video_processor.py:process_video()` | Resets sequence buffer for new video |
| `TemporalClassifier.extract_and_add_frame()` | ~60+ | `video_processor.py:process_video()` | Extracts M1 features and adds to sequence buffer |
| `TemporalClassifier.has_enough_frames()` | ~60+ | `video_processor.py:process_video()` | Checks if sequence buffer has enough frames for classification |
| `TemporalClassifier.classify_sequence()` | ~60+ | `video_processor.py:process_video()` | Classifies the sequence, returns P(accident) |

### `RapidAid-Accident-Detection-System/models/accident_zone.py`
Primary responsibility: Computes tight bounding zone around accident area from involved vehicles + victims
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `AccidentZoneCalculator.compute()` | ~23 | `frame_processor.py:process()` | Computes accident zone bbox from involved vehicles and victims |

### `RapidAid-Accident-Detection-System/models/vehicle_pedestrian_detector.py`
Primary responsibility: Conservative vehicle-pedestrian collision detection using geometry + posture
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `VehiclePedestrianDetector.detect()` | ~60+ | `frame_processor.py:process()` | Detects VP collisions, returns list of collision dicts |

### `RapidAid-Accident-Detection-System/utils/geometry.py`
Primary responsibility: Core geometric functions — IoU, overlap, distance, bbox operations
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `compute_iou()` | ~12 | `accident_classifier.py`, `collision_detector.py`, `vehicle_detector.py` | IoU between two bboxes |
| `compute_overlap_ratio()` | ~42 | `frame_processor.py`, `victim_classifier.py`, `vehicle_pedestrian_detector.py` | Fraction of smaller box overlapped by larger |
| `compute_edge_distance()` | ~73 | `frame_processor.py`, `accident_classifier.py`, `vehicle_pedestrian_detector.py` | Minimum edge-to-edge distance between bboxes |
| `compute_diagonal()` | ~113 | `accident_classifier.py`, `accident_zone.py` | Diagonal length of bbox |
| `compute_box_area()` | ~118 | `vehicle_detector.py`, `person_detector.py`, `accident_zone.py` | Area of bbox |
| `compute_box_center()` | ~123 | `frame_processor.py`, `collision_detector.py`, `accident_zone.py` | Center point of bbox |
| `compute_aspect_ratio()` | ~130 | `person_detector.py`, `accident_classifier.py` | Width/height ratio |
| `check_pixel_collision()` | ~139 | `accident_classifier.py` | Pixel-level polygon overlap |
| `compute_relative_angle()` | ~175 | `accident_classifier.py` | Angle between two bbox centers |
| `point_in_box()` | ~193 | `victim_classifier.py` | Point-in-bbox test |
| `expand_box()` | ~207 | `accident_zone.py` | Expand bbox by padding |
| `merge_boxes()` | ~229 | `accident_zone.py` | Enclosing bbox for all input boxes |

### `RapidAid-Accident-Detection-System/utils/helpers.py`
Primary responsibility: File I/O, timestamp generation, frame saving utilities
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `get_timestamp()` | ~13 | `pipeline/report_generator.py` | Returns current timestamp string |
| `save_annotated_frame()` | ~18 | `pipeline/report_generator.py:save()` | Saves annotated frame to outputs directory |
| `save_report()` | ~38 | `pipeline/report_generator.py:save()` | Saves JSON report to outputs directory |
| `load_frame()` | ~59 | `main.py` | Loads frame from file path |
| `display_frame()` | ~80+ | `video_processor.py`, `main.py` | Displays frame in OpenCV window |

### `RapidAid-Accident-Detection-System/utils/visualization.py`
Primary responsibility: Frame annotation — drawing vehicles, victims, zones, banners
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `annotate_frame()` | ~80+ | `pipeline/report_generator.py:generate()` | Full annotation: vehicles + victims + zone + banner |
| `draw_banner()` | ~15 | `annotate_frame()` | Draws status banner at top of frame |
| `draw_vehicle()` | ~40 | `annotate_frame()` | Draws vehicle bbox + polygon + label |

### `RapidAid-Accident-Detection-System/config/settings.py`
Primary responsibility: Single source of truth for all thresholds, paths, and configuration constants
| Setting Group | Keys | Used By |
|--------------|------|---------|
| PATHS | `PROJECT_ROOT`, `WEIGHTS_DIR`, `OUTPUTS_DIR`, `ANNOTATED_DIR`, `REPORTS_DIR`, `DATA_DIR` | All modules |
| MODEL PATHS | `VEHICLE_MODEL`, `POSE_MODEL`, `ACCIDENT_CLASSIFIER_MODEL`, `DAMAGE_CLASSIFIER_MODEL`, `TEMPORAL_CLASSIFIER_MODEL` | `frame_processor.py`, `track_processor.py`, model init |
| VEHICLE DETECTION | `VEHICLE_CONFIDENCE_THRESHOLD`, `MIN_VEHICLE_AREA_RATIO`, `MAX_VEHICLE_AREA_RATIO` | `vehicle_detector.py`, `track_processor.py` |
| ACCIDENT CLASSIFICATION | `MASK_OVERLAP_THRESHOLD`, `BBOX_IOU_THRESHOLD`, `CRASH_SCORE_THRESHOLD`, `MAX_CRASH_SCORE`, `SINGLE_VEHICLE_CLASSIFIER_THRESHOLD`, `SCENE_ONLY_CLASSIFIER_THRESHOLD` | `accident_classifier.py`, `frame_processor.py` |
| M2 TRUST | `M2_TRUST_ENABLED`, `DAMAGE_ONLY_MIN_SCENE_CONF`, `DAMAGE_ONLY_MIN_DAMAGE` | `frame_processor.py` |
| VICTIM | `PERSON_CONFIDENCE_THRESHOLD`, `STANDING_RATIO`, `LYING_RATIO`, `VICTIM_ONLY_MIN_COUNT`, `VEHICLE_VICTIM_DISTANCE` | `victim_classifier.py`, `frame_processor.py` |
| ZONE | `ZONE_PADDING_RATIO`, `MIN_ZONE_SIZE_RATIO`, `MAX_ZONE_SIZE_RATIO` | `accident_zone.py` |
| VIDEO | `FRAMES_PER_SECOND_TO_ANALYZE`, `POST_CRASH_LOOKAHEAD_SEC`, `TEMPORAL_CONFIRM_COUNT`, `TEMPORAL_WINDOW_SIZE` | `video_processor.py`, `track_processor.py` |
| PORTRAIT | `PORTRAIT_GEOMETRY_ENABLED` | `track_processor.py:_compute_geometry_score()` |
| ORCHESTRATION | `ANCHOR_DIFF_MIN_SEC` (1.0), `IMPACT_WINDOW_HALF_SEC`, `IMPACT_FALLBACK_HALF_SEC`, `MIN_STATE_SEP_SEC` | `frame_selector.py`, `event_extractor.py` |
| PHASE 6 IMPACT ZONE | `IMPACT_ZONE_PRE_SEC` (1.5), `IMPACT_ZONE_POST_SEC` (2.5) | `frame_selector.py:get_impact_zone()`, `_select_indices()` |

| BAKLLAVA | `BAKLLAVA_TIMEOUT_SEC`, `BAKLLAVA_SHORT_SIDE_PX`, `BAKLLAVA_MAX_LONG_SIDE_PX`, `BAKLLAVA_MIN_NARRATION_CHARS` | `bakllava_client.py` (fallback only) |
| FLORENCE-2 | `SEMANTIC_CLIENT` ('florence'/'bakllava'), `FLORENCE_MODEL_ID` (local weights/florence2/), `FLORENCE_MAX_NEW_TOKENS` (200), `FLORENCE_NUM_BEAMS` (3) | `florence_client.py`, `pipeline_manager.py:_init_semantic_client()` |
| ENSEMBLE | `ENSEMBLE_WEIGHT_GEOMETRIC`, `ENSEMBLE_WEIGHT_SCENE`, `ENSEMBLE_WEIGHT_DAMAGE` | `frame_processor.py` |
| M3 TEMPORAL | `TEMPORAL_SCORE_WEIGHT`, `TEMPORAL_SCORE_THRESHOLD` | `video_processor.py` |

### `RapidAid-Accident-Detection-System/config/vehicle_classes.py`
Primary responsibility: COCO-to-RapidAid vehicle class mappings and display names
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `is_vehicle_class()` | ~60 | `track_processor.py:__init__()` | Returns True if COCO class ID is a vehicle |
| `get_rapidaid_label()` | ~50 | `vehicle_detector.py` | Converts COCO class ID to RapidAid label |
| `get_display_name()` | ~55 | `vehicle_detector.py` | Gets human-readable vehicle name |

### `RapidAid-Accident-Detection-System/main.py`
Primary responsibility: CLI entry point for image and video processing
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `process_image()` | ~30 | CLI `--image` flag | Processes single image through FrameProcessor |
| `process_video()` | ~60+ | CLI `--video` flag | Processes video through VideoProcessor |

---

## Orchestration Layer

### `orchestration/pipeline_manager.py`
Primary responsibility: End-to-end multi-stage event validation orchestrator
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `PipelineManager.__init__()` | ~39 | `main_pipeline.py` | Initializes FrameSelector, StoryboardGenerator, TimelineBuilder, MetadataPackager, ConsensusEngine; calls `_init_semantic_client()` |
| `PipelineManager._init_semantic_client()` | ~55 | `__init__()` | Selects semantic client: `FlorenceClient` if `settings.SEMANTIC_CLIENT=='florence'`, else `bakllava_client`. Sets `self._semantic_client`. |
| `PipelineManager.process_video()` | ~60 | `main_pipeline.py` | Full 10-stage pipeline: RapidAid → extract → storyboard → Florence-2/bakllava → Groq → consensus → tier → reports |
| `PipelineManager._get_rapidaid()` | ~53 | `process_video()` | Lazy-loads TrackCentricProcessor |
| `PipelineManager._extract_clip()` | ~297 | `process_video()` | Extracts event clip using EventExtractor |
| `PipelineManager._collect_key_frames()` | ~319 | `process_video()` | Re-reads video to collect frames needed by FrameSelector |
| `PipelineManager._copy_to_tier()` | ~378 | `process_video()` | Copies key files to tier directory |
| `PipelineManager._save_event()` | ~404 | `process_video()` | Saves full event result as JSON |
| `PipelineManager._save_debug_reasoning()` | ~416 | `process_video()` | Saves debug_reasoning.md with diagnostic info |

### `orchestration/frame_selector.py`
Primary responsibility: Selects 5 event-state frames using causal disruption scoring
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `_causal_disruption_score()` | ~30 | `_select_indices()` | Computes composite disruption score: vel*0.30 + flow*0.25 + geo*0.20 + dis*0.15 + deaths*0.10 |
| `get_impact_zone()` | ~79 | unit tests / external callers | Phase 6: computes impact zone from first_confirmed_time (or event_time fallback). Returns (zone_frames, source). source in {'first_confirmed', 'event_time', + '_expanded'/'_fallback' suffixes} |
| `FrameSelector.select_storyboard_frames()` | ~143 | `pipeline_manager.py:process_video()` | Selects 5 event-state frames with roles, timestamps, frame_data |
| `FrameSelector.get_needed_frame_indices()` | ~450+ | `pipeline_manager.py:_collect_key_frames()` | Pre-computes which analyzed frame indices to extract |
| `FrameSelector._select_indices()` | ~210+ | `select_storyboard_frames()` | Phase 6: selects indices using anchor_time with IMPACT_ZONE_PRE/POST_SEC window (no GT lookup) |
| `FrameSelector._find_impact_index()` | ~370+ | `_select_indices()` | Selects impact frame using earliest strong disruption within impact_window |
| `FrameSelector._find_event_state_indices()` | ~280+ | `_select_indices()` | Phase 6: selects pre/convergence/impact/peak/aftermath indices -- no GT branch |
| `FrameSelector._enforce_min_separation()` | ~250+ | `_find_event_state_indices()` | Ensures minimum temporal gap between selected frames |
| `FrameSelector._deduplicate_indices()` | ~430+ | `_find_event_state_indices()` | Ensures all 5 indices are distinct and strictly increasing |

Phase 6 changes:
- REMOVED: `_normalize_video_name()`, `_get_gt_window()` (GT lookup -- gone)
- REMOVED: `from shared.constants import GT_IMPACT_WINDOWS` import
- REMOVED: 'gt_supervised' source string -- no longer generated
- ADDED: `get_impact_zone()` module-level function for testable zone computation
- impact_zone_source values: 'first_confirmed' | 'event_time' | + '_expanded'/'_fallback'


### `orchestration/event_extractor.py`
Primary responsibility: Rolling frame buffer + event clip extraction (T-8s to T+12s)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `EventExtractor.__init__()` | ~21 | `pipeline_manager.py:_extract_clip()` | Initializes deque buffer with max_frames based on fps |
| `EventExtractor.add_frame()` | ~29 | `pipeline_manager.py:_extract_clip()` | Adds frame to rolling buffer |
| `EventExtractor.extract_event_clip()` | ~38 | `pipeline_manager.py:_extract_clip()` | Extracts event clip, saves raw + overlay MP4s + key frames |

### `orchestration/consensus_engine.py`
Primary responsibility: Weighted fusion tier assignment with physics safeguard
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `ConsensusEngine.__init__()` | ~31 | `pipeline_manager.py:__init__()` | Initializes major/minor confidence thresholds |
| `ConsensusEngine.evaluate()` | ~39 | `pipeline_manager.py:process_video()` | Evaluates all evidence, assigns tier with physics safeguard |
| `ConsensusEngine._count_strong_signals()` | ~221 | `evaluate()` | Counts causal signal families exceeding threshold |
| `ConsensusEngine.save_consensus()` | ~237 | `pipeline_manager.py:process_video()` | Saves consensus result to JSON |

### `orchestration/metadata_packager.py`
Primary responsibility: Packages RapidAid signals + timeline + narration for Groq consumption
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `MetadataPackager.package()` | ~18 | `pipeline_manager.py:process_video()` | Creates comprehensive metadata package with causal data |
| `MetadataPackager._extract_impact_signals()` | ~134 | `package()` | Extracts detailed causal signals at impact frame |
| `MetadataPackager._signals_from_frame_data()` | ~151 | `_extract_impact_signals()` | Extracts signal dict from single frame's data |
| `MetadataPackager._extract_event_state_labels()` | ~177 | `package()` | Extracts role labels and timestamps for Groq |
| `MetadataPackager._extract_confidence_curve()` | ~249 | `package()` | Sparse confidence progression for temporal context |
| `MetadataPackager._extract_track_lifecycle()` | ~270 | `package()` | Track count evolution for Groq's temporal reasoning |
| `MetadataPackager.save_package()` | ~292 | `pipeline_manager.py:process_video()` | Saves metadata package to JSON |

### `orchestration/timeline_builder.py`
Primary responsibility: Constructs structured event timelines from RapidAid metadata
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `TimelineBuilder.build_timeline()` | ~15 | `pipeline_manager.py:process_video()` | Builds timeline with phases, impact signals, metrics |
| `TimelineBuilder.save_timeline()` | ~92 | `pipeline_manager.py:process_video()` | Saves timeline to JSON |

### `orchestration/storyboard_generator.py`
Primary responsibility: Saves 5 individual high-resolution event-state frames + composite
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `StoryboardGenerator.generate()` | ~66 | `pipeline_manager.py:process_video()` | Generates individual frames (clean + debug) + composite image |
| `StoryboardGenerator._build_composite()` | ~198 | `generate()` | Builds horizontal strip composite for human review |
| `StoryboardGenerator._resize_with_letterbox()` | ~221 | `generate()` | Resizes frame preserving aspect ratio with letterboxing |

### `src/florence_client.py`
Primary responsibility: Per-frame temporal event narration using Florence-2-large (Phase 7, primary engine)
**GPU:** RTX 3050 float16, 1.55GB VRAM, 2.0s load, 0.81s/frame inference (3-beam, 200 tok)
**Weights:** `weights/florence2/` (local, not in git). See `settings.FLORENCE_MODEL_ID`.
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `FlorenceClient.__init__()` | ~65 | `pipeline_manager.py:_init_semantic_client()` | Sets device/dtype, calls `_load_model()` |
| `FlorenceClient._load_model()` | ~74 | `__init__()` | Loads processor + model from local path using `.to(device)` (device_map unsupported). `local_files_only=True`. |
| `FlorenceClient.is_available` | ~100 | `pipeline_manager.py` | Property: True if model loaded successfully |
| `FlorenceClient.narrate_frame()` | ~145 | `narrate_event()`, unit tests | Single-frame narration: BGR→PIL, task token only prompt, prepends role_ctx to output |
| `FlorenceClient.narrate_event()` | ~200 | `pipeline_manager.py:process_video()` | Narrates all event-state frames, returns `{role_label: narration}` dict |
| `FlorenceClient._bgr_to_pil()` | ~135 | `narrate_frame()` | Converts OpenCV BGR ndarray to RGB PIL Image |
| `FlorenceClient._extract_narration()` | ~190 | `narrate_frame()` | Extracts plain text from Florence-2 post-processed output dict |

**Critical notes:**
- `device_map='auto'` NOT supported — use `.to(device)` instead
- Task token (`<DETAILED_CAPTION>`) must be the ONLY input text — role context is prepended to the OUTPUT, not the prompt
- `MODEL_ID = settings.FLORENCE_MODEL_ID` (class attribute reads from settings at import time)

### `src/bakllava_client.py`
Primary responsibility: Per-frame temporal event narration using bakllava (Ollama local) — **FALLBACK only (Phase 7+)**
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `preprocess_for_bakllava()` | ~82 | `_send_single_frame()` | Rotates portrait frames, normalizes size for bakllava |
| `mask_timestamp_region()` | ~111 | `_send_single_frame()` | Blacks out CCTV overlay regions to prevent timestamp reading |
| `narrate_event_frames()` | ~144 | `narrate_storyboard()` | Sends 5 independent frames to bakllava, runs synthesis pass |
| `narrate_storyboard()` | ~193 | `pipeline_manager.py:process_video()` | Backward-compatible entry point for narration |
| `_send_single_frame()` | ~214 | `narrate_event_frames()` | Sends single frame to bakllava with prompt |
| `_run_synthesis()` | ~253 | `narrate_event_frames()` | Text-only synthesis pass combining frame narrations |
| `check_bakllava_available()` | ~318 | `pipeline_manager.py:process_video()` | Checks if bakllava model is available in Ollama |

### `src/groq_reasoner.py`
Primary responsibility: Structured JSON synthesis using Groq cloud (llama-3.1-8b-instant)
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `synthesize()` | ~86 | `pipeline_manager.py:process_video()` | Runs Groq synthesis on full metadata package, returns structured assessment |
| `_build_prompt()` | ~136 | `synthesize()` | Builds complete synthesis prompt with all evidence sections |
| `_fallback_result()` | ~223 | `synthesize()` | Generates fallback when Groq is unavailable |

### `src/config.py`
Primary responsibility: Centralized configuration — API URLs, keys, paths, logging
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `generate_random_gps()` | ~27 | `report_generator.py` | Generates random GPS coordinates within radius |
| `setup_logging()` | ~73 | `main.py` | Configures application logging (file + console) |

### `src/main.py`
Primary responsibility: Main orchestration for image-based accident report generation
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `generate_accident_report()` | ~21 | `app.py` (Streamlit) | Full pipeline: load → detect → analyze → report → PDF |

### `src/image_handler.py`
Primary responsibility: Image loading, validation, preprocessing for API/PDF
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `validate_image()` | ~17 | `load_image()` | Validates image exists and is supported format |
| `load_image()` | ~43 | `main.py:generate_accident_report()` | Loads PIL Image from file path |
| `preprocess_image()` | ~60+ | `ollama_client.py` | Preprocesses image for API consumption |
| `image_to_base64()` | ~60+ | `ollama_client.py` | Converts PIL Image to base64 string |

### `src/report_generator.py`
Primary responsibility: Generates structured accident reports from AI model responses
| Function | Line | Called By | Does |
|----------|------|-----------|------|
| `parse_structured_fields()` | ~20 | `ollama_client.py` | Parses JSON from model response with robust error handling |
| `create_report_structure()` | ~60+ | `main.py:generate_accident_report()` | Creates report dict from parsed fields |
| `format_accident_report()` | ~60+ | `main.py:generate_accident_report()` | Formats report as human-readable text |
| `generate_report_filename()` | ~60+ | `main.py:generate_accident_report()` | Generates timestamped report filename |

### `shared/constants.py`
Primary responsibility: Event tier definitions, project paths, ground truth impact windows, consensus thresholds
| Constant/Class | Line | Called By | Does |
|----------|------|-----------|------|
| `PROJECT_ROOT`, `RAPIDAID_ROOT`, `REPORTS_ROOT`, `EVENTS_DIR` | ~8-11 | All orchestration modules | Project directory paths |
| `TIER_DIRS` | ~14 | `pipeline_manager.py` | Maps tier names to report directories |
| `EventTier` | ~28 | `consensus_engine.py`, `pipeline_manager.py` | Tier constants: VERIFIED_MAJOR, VERIFIED_MINOR, AMBIGUOUS, LOW_CONFIDENCE |
| `GT_IMPACT_WINDOWS` | ~45 | COMMENTED OUT (Phase 6) | Ground truth impact windows -- reference only, not imported anywhere. All lines preceded by '#'. |

| `EVENT_PRE_SECONDS`, `EVENT_POST_SECONDS` | ~56-57 | `event_extractor.py` | Clip extraction time buffer (8s pre, 12s post) |
| `CONSENSUS_VETO_KEYWORDS` | ~64 | `consensus_engine.py` | Keywords that trigger semantic veto check |

---

## Signal Flow Summary
```
[input video]
    → track_processor.py        (YOLO detection + ByteTrack + signals)
    → confidence_fusion.py      (fused confidence score)
    → pipeline_manager.py       (orchestration)
    → frame_selector.py         (5 storyboard frames)
    → florence_client.py        (narration — PRIMARY, GPU, local weights)
       [or bakllava_client.py]  (narration — FALLBACK, requires Ollama)
    → groq_reasoner.py          (synthesis)
    → consensus_engine.py       (tier decision)
    → metadata_packager.py      (output JSON)
```

Client selected by `settings.SEMANTIC_CLIENT`: `'florence'` (default) or `'bakllava'`.

## Known Fragile Points

1. **`consensus_engine.py:evaluate()`** — Changing `physics_overwhelming` or `physics_strong` thresholds affects ALL tier assignments. Used by `pipeline_manager.py`.

2. **`frame_selector.py:_causal_disruption_score()`** — Weight changes (vel×0.30, flow×0.25, etc.) affect which frames are selected for semantic analysis, cascading to bakllava + Groq outputs.

3. **`track_processor.py:process_video()`** — The main fusion loop. Changes to EMA smoothing (0.7/0.3), confidence thresholds, or state machine transitions affect every video processed.

4. **`causal_gate.py:evaluate()`** — State machine thresholds (`confirm_threshold=0.45`, `suspicious_threshold=0.30`, `confirm_frames_required=3`) determine when accidents are confirmed. Affects all downstream anchoring.

5. **`metadata_packager.py:package()`** — Output schema is consumed by both `groq_reasoner.py` and `consensus_engine.py`. Adding/removing fields breaks both callers.

6. **`settings.py` constants** — `ENSEMBLE_WEIGHT_*`, `M2_TRUST_ENABLED`, `CRASH_SCORE_WEIGHTS` are referenced by multiple files. Changes have wide blast radius.

7. **`bakllava_client.py:narrate_event_frames()`** — Fallback only (Phase 7+). Portrait frame rotation + timestamp masking affect narration quality when `settings.SEMANTIC_CLIENT='bakllava'`.

8. **`shared/constants.py:GT_IMPACT_WINDOWS`** — COMMENTED OUT Phase 6. No longer imported anywhere. Reference block preserved for Phase 3 training documentation.

9. **`florence_client.py:FlorenceClient`** — `MODEL_ID` is a class attribute read from `settings.FLORENCE_MODEL_ID` at import time. If `settings.py` or `weights/florence2/` is missing, the class silently sets `_model=None` and all narration returns `[florence: model unavailable]`. Task token must be sole prompt input — any prefix causes `ValueError`. `device_map='auto'` unsupported — must use `.to(device)`.
