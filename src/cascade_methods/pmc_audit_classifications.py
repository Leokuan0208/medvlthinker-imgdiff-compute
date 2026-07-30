#!/usr/bin/env python3
"""
pmc_audit_classifications.py -- the per-item label-quality verdicts for the PMC-VQA noise audit.

Every verdict below was reached by opening the actual figure image with the Read tool and reading it
next to the question, the four options, the gold letter, both models' raw answers, and the PMC-VQA
SOURCE CAPTION (PMC-VQA's gold was auto-generated from the caption, so the caption is the key's own
provenance). Writes results/cascade_methods/artifacts/_pmc_audit_classifications.json, which
pmc_label_noise_audit.py --stage score consumes.

RUBRIC (fixed before scoring; applied identically to wins, losses and the agree-correct control):
  GENUINE       the gold is correct/consistent with the source caption AND the answer is in principle
                derivable from the shown image by a domain expert -> the item can support an
                accuracy claim.
  BAD-GOLD      the keyed letter is wrong, contradicts the caption or the image, or the OTHER model's
                answer is at least as defensible -> the scored outcome is an artifact of the key.
  UNANSWERABLE  the answer is not in the shown image at all: the crop is a different panel/modality
                than the question refers to, the referenced marker (arrow/asterisk/inset/row/panel
                letter) is absent, the discriminating information is temporal or metadata, or the
                distinction is a caption-only convention.
  MULTI-CORRECT more than one option is defensibly correct (usually because the caption names two of
                them).
  UNCLEAR       I could not judge; recorded separately and never counted as a defect.
Precedence when several apply: BAD-GOLD > UNANSWERABLE > MULTI-CORRECT > GENUINE.
Deliberately conservative: a hard-but-well-posed expert call counts as GENUINE, and an awkwardly
worded but image-answerable question counts as GENUINE.
"""
import json, os

OUT = os.path.expanduser(
    "~/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/_pmc_audit_classifications.json")

# ------------------------------------------------------------------ WINS (method right / 32B wrong)
WINS = [
("pmc-151",   "GENUINE",       "Caption keys the bony IAC roof to '[green lines]'; the HRCT panel's only non-yellow annotation colour is green (no red/blue/purple present), so the key is right and image-derivable."),
("pmc-278",   "BAD-GOLD",      "Caption: 'large ILL-DEFINED, heterogeneous mass'. Gold B is 'Large, defined mass' while option D is verbatim 'Large, ill-defined mass'. The 32B answered D and was scored wrong; the key is wrong."),
("pmc-415",   "UNANSWERABLE",  "Asks which region 'was analyzed in the study'. Crop is one unmarked axial posterior-fossa slice; the answer lives in the paper text."),
("pmc-705",   "GENUINE",       "Axial chest CT shows a bulky left mediastinal mass encasing the airway; 'left main bronchus' matches the caption and is a legitimate expert read."),
("pmc-1006",  "GENUINE",       "Coronal CT: abdominal contents and vessels occupy the image-left (patient's right) hemithorax while the left lung is aerated. Gold 'Right' is correct and visible."),
("pmc-1632",  "MULTI-CORRECT", "Degraded right-hand sagittal panel reads equally as the caption's 'striping' (gold) or as blur (options B/C); the key rests on the caption's word choice."),
("pmc-2758",  "GENUINE",       "Coronary angiogram labelled 'Abnormal vessel'; gold (left circumflex) matches the caption and the vessel is expert-identifiable."),
("pmc-3417",  "UNANSWERABLE",  "Crop is an intra-operative gross photograph (panel D); the keyed 'peripheral ossification' is a CT finding from panels A-B. Caption also names three of the four options."),
("pmc-3803",  "UNANSWERABLE",  "Question asks the subject's condition (AVN of the femoral head per caption) but the crop is an H&E bone/marrow histology panel, not the hip MRI."),
("pmc-3939",  "BAD-GOLD",      "Caption puts the arrow on the sagittal PET/CT panel (d''); the crop is the axial CT panel (d') and contains no arrow. Gold 'Sagittal CT view' matches neither the caption nor the image."),
("pmc-3978",  "UNANSWERABLE",  "Question asks what light blue AND orange represent; only the orange hind-limb overlay is present in the crop, the light-blue lung-tumour segmentation is absent."),
("pmc-4232",  "GENUINE",       "Caption-referential wording ('mentioned in the caption'), but the coronal T1 wrist MR with the arrow shows the scaphoid; the key is correct and image-derivable."),
("pmc-4666",  "BAD-GOLD",      "'ADC PET' does not exist (ADC is a diffusion-MRI map). Caption reads '(D) ADC and NAC PET', so the key is a caption-parsing error and none of FAC/DAC/ADC/RAC PET is a real modality."),
("pmc-5044",  "UNANSWERABLE",  "Non-medical (nemertean worm). Crop is panel 'n' while the caption's referent is panel (e); anterior/mid/posterior body is not judgeable from a cropped micrograph."),
("pmc-5133",  "GENUINE",       "Bronchoscopic view shows the yellow endobronchial blocker in the airway; gold C matches and the CT-only distractor is excludable."),
("pmc-5134",  "BAD-GOLD",      "Caption states the sampling was TRANSPERINEAL ('inferior quality of transperineal US compared to transrectal'); gold 'Transrectal' inverts it. The 32B's 'transperineal' is better supported."),
("pmc-5729",  "UNANSWERABLE",  "Non-medical (insect antennal-lobe glomeruli). Neither species nor sex is inferable from the numbered confocal reconstructions."),
("pmc-6061",  "GENUINE",       "Sagittal breast MR shows a clearly hypointense spiculated mass against bright surrounding tissue; gold 'Hypointense' matches."),
("pmc-6372",  "UNANSWERABLE",  "Non-medical (steel carbide TEM). No circle is present in the crop and 'with/without a high density of carbides' is not judgeable."),
("pmc-6414",  "GENUINE",       "The open arrow visible in the axial CT-arthrogram sits on the palmar/volar scapholunate ligament; gold 'Volar' matches and no other option is in the caption."),
("pmc-6417",  "GENUINE",       "The arrowhead is on the dorsal aspect of the axial wrist image (dorsal is up); gold 'dorsal part of SLL' matches the caption."),
("pmc-6912",  "UNANSWERABLE",  "Nodal-region descriptor (left internal iliac / obturator fossa) with four near-identical options; the crop is a small dark MR panel and the caption's referent is HE histology."),
("pmc-6937",  "UNANSWERABLE",  "Non-medical (fossil tooth microCT surface render). 'Posterior view' is a figure convention, not visible content."),
("pmc-7558",  "UNANSWERABLE",  "Pure panel cross-reference ('which panel shows the scans as in panel e'); the crop is a single BAF fundus image with no panel letters."),
("pmc-8465",  "MULTI-CORRECT", "Caption annotates pleomorphic nuclei, mitotic nuclei AND a prominent myxoid matrix - three of the four options. The question asks for the feature of the tumour CELLS, so the 32B's 'pleomorphic nuclei' is defensible."),
("pmc-8720",  "UNANSWERABLE",  "Single lateral hip/femur radiograph with no side marker; the question refers to 'the view on the right' of a two-panel figure that is not shown."),
("pmc-9402",  "GENUINE",       "Gold 'T-1 FLAIR' matches the caption and MRI sequence identification is a legitimate image task (option A 'T-3 FLAIR' is not a sequence)."),
("pmc-9577",  "UNANSWERABLE",  "The electron-micrograph panels (D-G) carrying the red arrowheads are almost entirely cropped out; only a sliver is visible."),
("pmc-9666",  "GENUINE",       "The crop is a plain radiograph with an 'L' marker; 'Supine anteroposterior' is the only x-ray option (CT/MRI/PET excluded by inspection)."),
("pmc-10497", "UNANSWERABLE",  "Non-medical (cryo-FIB lamella). Markers in the crop are white brackets, and 'micro-expansion joints' vs 'cracks in the lamella' is a caption-only distinction."),
("pmc-11571", "GENUINE",       "H&E shows gastrointestinal wall (crypts, lamina propria, smooth-muscle band); gold B is the only histologically possible option."),
("pmc-11838", "UNANSWERABLE",  "Crop is a photograph of the physical phantom on the CT couch, not the CT slice; the rows of low-contrast spheres are not visible."),
("pmc-11950", "GENUINE",       "Fused PET/CT shows hypermetabolic breast/chest-wall foci; the two 'decreased uptake' options are self-excluding and no arm uptake is shown."),
("pmc-12865", "GENUINE",       "H&E panel shows liver parenchyma with a metastatic deposit and a lymphoid infiltrate; 'Liver' is expert-derivable from the tissue."),
("pmc-13058", "UNANSWERABLE",  "Severe image/caption mismatch: the question asks what a BLUE arrow indicates on a head CT showing hydrocephalus, but the crop is a SPLEEN ultrasound with a yellow arrow. The 7B's win is pure option-text prior."),
("pmc-13076", "GENUINE",       "Axial posterior-fossa MR shows abnormal vermian/cerebellar signal; gold 'lesion in the vermis' matches the caption and is visible."),
("pmc-13172", "GENUINE",       "Correctly keyed, but NON-VISUAL: 'What does PET-CT stand for?' is acronym expansion and measures no visual ability."),
("pmc-13280", "UNANSWERABLE",  "The crop (panel C, dark image with a red dotted ellipse) contains no white cross; the conventional-US panel the question refers to is absent."),
("pmc-13674", "BAD-GOLD",      "The scanner header burned into the image reads 'RLD' (right lateral decubitus); gold says 'Supine'. The 32B answered 'Right Lateral' - matching the image - and was scored wrong."),
("pmc-13685", "GENUINE",       "Angiogram shows the classic RCA course to the posterior descending artery; gold 'Right coronary artery' is correct and the 32B's 'LAD' is not."),
("pmc-13734", "GENUINE",       "Axial head-CT bone window with a lytic calvarial/temporal defect; the mandible/maxilla, cervical-spine and pelvis options are excluded by the level shown."),
("pmc-13904", "GENUINE",       "Crop is a fat-saturated T2 neck MR with a parotid mass and an adjacent node; the competing options name modalities that are not shown."),
("pmc-14795", "GENUINE",       "Sagittal hindfoot CT with arrowheads on lucent intraosseous areas; gold 'low-attenuation areas' matches."),
("pmc-14901", "GENUINE",       "AP hip radiograph; laterality is derivable from the pelvic-anatomy orientation and gold 'left hip' matches the caption."),
("pmc-14914", "GENUINE",       "Axial contrast CT with a stent graft; 'level of the origin of the left pulmonary artery' matches the caption and is a legitimate (if hard) expert call."),
("pmc-15131", "UNANSWERABLE",  "No white arrow anywhere in the crop (a colour-segmented panel with crosshairs); the marker the question depends on is absent."),
("pmc-15185", "BAD-GOLD",      "Brown tumours are a manifestation of hyperPARAthyroidism, not Paget's disease; the caption describes focal neck uptake inferior to the thyroid (parathyroid). Gold 'Paget's disease' is medically wrong and no option is correct. Crop is two near-black scintigraphy panels."),
("pmc-15633", "GENUINE",       "A 2D slice plus a 3D reconstruction of a bone specimen is microCT; 'CT' is the best option and matches the caption."),
("pmc-15643", "GENUINE",       "Fat-suppressed high-signal lobular lesion is consistent with the caption's T2-weighted sequence; gold C matches."),
("pmc-15988", "GENUINE",       "Crop is a bright-field micrograph of a gut cross-section; gold A is the only fitting option (C names the cecum, a different panel)."),
("pmc-16442", "UNANSWERABLE",  "Mass mobility ('Moving' vs 'Fixed') cannot be determined from a still echocardiographic frame, and the left atrium is barely inside the crop."),
("pmc-16518", "UNCLEAR",       "3D CT ankle reconstruction; I cannot make out fixation hardware in the render, so I cannot tell whether the plated bone is visible or whether this is a pre-operative study."),
("pmc-16541", "UNANSWERABLE",  "The referenced inset 'Bi' is present only as an empty marker box; its content is not in the crop, and basement membrane vs endothelial projection is not resolvable at this scale."),
("pmc-16959", "GENUINE",       "Crop is unmistakably a 3D brain rendering with tractography; correctly keyed but DEGENERATE (distractors are 'a flower', 'a painting', 'a car')."),
("pmc-17207", "UNANSWERABLE",  "Non-medical (shale thin sections). One low-resolution featureless panel; the caption's multi-panel referent is absent and gold 'calcite shale' does not even match the caption's 'calcareous shale'."),
("pmc-17498", "UNANSWERABLE",  "Non-medical (plant-fossil paratype). Several white arrows are present, and basalmost vs distalmost leaf attachment is a palaeobotany call."),
("pmc-17635", "MULTI-CORRECT", "Caption lists anomalies at C1-C7 and T8; option A (C2 and C4, both in the listed fusions) and option C (T8) are also supported, so gold B (C3 and C5) is not uniquely correct."),
("pmc-17854", "MULTI-CORRECT", "Caption names BOTH 'right frontal AND left parietofrontal' regions, so the 32B's 'left frontal' is defensible; the crop is also an MR, not the plain CT the question names."),
("pmc-18013", "BAD-GOLD",      "Crop is a 2D slice carrying the caption's '2D identification points'; the '3D volumetric' joint space is a different panel. The 32B's '2D joint space' is at least as defensible for the image shown."),
("pmc-18066", "GENUINE",       "Crop is a PET of the two lower limbs; gold 'Lower body' matches and is obvious."),
("pmc-18278", "GENUINE",       "H&E shows woven bone/new-bone formation labelled 'C'; gold 'Callus formation' matches the caption and the label is in the image."),
("pmc-19363", "UNANSWERABLE",  "Panel referent is inconsistent (crop is a nasopharynx-level CT labelled 'B' while the caption's B is a thorax image) and a green arrow marks a finding, contradicting the 'no enhancing lesion' key. Options C/D are degenerate."),
("pmc-19599", "UNANSWERABLE",  "Single-panel crop; the question asks which ROW the caption 'describes' as lacking deep-white-matter change, and no rows are present."),
("pmc-19936", "GENUINE",       "A thin white arrow marks a pancreatic cystic/ductal neoplasm (IPMN); 'Cancer' is the only neoplasm option. Imprecise (IPMN is premalignant) but correctly keyed."),
("pmc-21500", "UNCLEAR",       "'The plane represented by the 3D printed model' is ill-defined phrasing and I cannot verify left atrium vs left ventricle from the rendering."),
("pmc-21762", "BAD-GOLD",      "Caption's diagnosis for Case 51 (C) is invasive perilimbal SCC, which is not among the options; gold 'Papilloma' is lifted from a different panel's sentence."),
("pmc-21934", "GENUINE",       "Axial chest CT shows bilateral multilobar ground-glass opacity and consolidation; gold B matches."),
("pmc-22131", "GENUINE",       "Cartilage ultrasound panels labelled LFC(ant)/LFC(post); fracture and cancer are excludable and gold 'Degenerative Lesions' matches the caption."),
("pmc-22445", "GENUINE",       "Correctly keyed, but NON-VISUAL: the question states the finding (retinal hemangioblastomas) and asks the associated syndrome - textbook VHL knowledge, no image needed."),
("pmc-22976", "UNANSWERABLE",  "Only one section (panel f) is in the crop, so 'All of the presented sections' cannot be verified."),
("pmc-23337", "UNANSWERABLE",  "Non-medical (hydroxyapatite/collagen tomography). 'The z-slices indicate the location of...' is a caption statement, not derivable from the two tiny panels."),
("pmc-23769", "MULTI-CORRECT", "The shown axial fused PET/CT is at a liver/stomach level where 'lower thoracic' vs 'abdominal' aorta is genuinely ambiguous; the 32B's 'abdominal aorta' is defensible."),
("pmc-24120", "UNANSWERABLE",  "Image/caption mismatch: the caption describes the right femur medullary cavity but the crop is an axial CHEST CT with a red arrow. No option matches the shown anatomy; the 7B's 'femur' is pure text prior."),
("pmc-24199", "GENUINE",       "Brain MR crop; 'Brain' is the only anatomically possible option (distractors include 'small intestine'). Correctly keyed but degenerate and caption-referential."),
("pmc-24496", "UNANSWERABLE",  "An arrow cannot depict 'field intensity and rotation direction'; the semantics are caption-only and option B (fluid flow) is equally arrow-compatible."),
("pmc-24560", "GENUINE",       "Axial chest CT shows a right-sided perihilar/upper-lobe mass; gold 'Right upper lobe' matches the caption and the side is unambiguous."),
("pmc-24797", "GENUINE",       "Crop is a digital-subtraction angiogram with a stent - an x-ray technique; 'X-Ray' is the only defensible option among CT/MRI/X-ray/echo."),
("pmc-24810", "UNANSWERABLE",  "Severe mismatch: the question asks about blue labelling in photomicrographs, but the crop is a clinical photograph of a cat's ulcerated facial tumour. No blue label exists."),
("pmc-25045", "UNANSWERABLE",  "Only the T1w panel is shown; distinguishing early from late subacute haematoma requires paired T1 and T2 signal."),
("pmc-25102", "GENUINE",       "Crop shows x-ray images of maize seeds at four stages; gold 'X-ray images of maize seeds' matches. Correctly keyed but NON-MEDICAL."),
("pmc-25218", "GENUINE",       "MicroCT of a metaphysis with growth plate and dense trabeculae is the distal femur; shaft/head options are excluded by the image."),
("pmc-25510", "BAD-GOLD",      "Crop is panel C (adrenal metastasis, red arrow) but the question asks about Image A, which is absent. The 32B's 'Adrenal Metastasis' is correct FOR THE IMAGE SHOWN and was scored wrong."),
("pmc-25843", "UNANSWERABLE",  "Crop is a pseudo-coloured EM with three coloured processes, not the green/red fluorescence image the caption describes; axon vs dendrite is not determinable and 'two' is contradicted."),
("pmc-26032", "GENUINE",       "3D left-atrial view with the defect outlined in the mid interatrial septum plus SVC/Ao and orientation labels; secundum ASD is an expert-determinable call matching the caption."),
("pmc-26761", "MULTI-CORRECT", "Caption says the right panels show the cross section at BOTH low and higher magnification, so option A is as supported as gold B; the crop also does not show the panel layout."),
("pmc-26903", "UNANSWERABLE",  "The question asks what the CT revealed; the crop is the plain chest radiograph panel. The CT panel is absent."),
("pmc-27116", "GENUINE",       "The white arrow marks a pleural-based nodule in the left lower lobe (image-right, posterior); gold B matches the caption."),
("pmc-27180", "UNANSWERABLE",  "Crop is a contrast brain MR with no outline of any colour; the outlined H&E section the question refers to is absent."),
("pmc-27403", "GENUINE",       "Axial mandibular CBCT with a red ellipse over the image-left (patient's right) body spanning several tooth positions; gold 'Right premolar to first molar' matches."),
("pmc-27583", "UNANSWERABLE",  "Non-medical (algorithm residual maps). Single circular panel; the row structure the question depends on is absent."),
("pmc-27900", "UNANSWERABLE",  "Crop shows lines C/D, not the referenced 'Line A'; and SWI vs T2* (options B and A) is not reliably separable by eye in any case."),
("pmc-28151", "GENUINE",       "Mammogram with a thick white arrow on the tumour and a separate black arrowhead on the microcalcifications; gold 'Tumor site' matches and the markers disambiguate."),
("pmc-28314", "GENUINE",       "Main panel is unmistakably trichrome (green-stained collagen); gold 'Masson trichrome' matches the caption."),
("pmc-28863", "GENUINE",       "Coronal CT shows a small opacified image-left (patient's right) hemithorax with ipsilateral mediastinal shift; gold 'Right Lung' matches."),
("pmc-28909", "UNANSWERABLE",  "The caption distinguishes red dotted arrow / red arrowhead / red arrow across panels; the crop has one small red marker whose type is unresolvable at this resolution."),
("pmc-29475", "MULTI-CORRECT", "Crop shows a squamoid morule in endometrial glandular tissue - the caption's 'morular metaplasia in endometrioid carcinoma' - so option A is at least as defensible; 'adamantinomatous carcinoma' is also not a recognised entity."),
("pmc-29644", "UNANSWERABLE",  "Species (sheep vs boar/cow/llama) cannot be read off a lateral animal thoracic radiograph; the answer is in the caption."),
("pmc-29804", "GENUINE",       "Crop is a sagittal spine study; only option C ('stress fracture of the spinous process') is anatomically possible - 'thigh bone' and 'knee joint' are excluded."),
("pmc-30363", "GENUINE",       "The confocal panel is labelled CCe (corpus cerebelli) and Va (valvula cerebelli); 'cerebellum' matches and the labels are in the image."),
("pmc-31778", "UNANSWERABLE",  "Crop is the POST-operative instrumented sitting lateral film; the question asks about the PRE-operative condition. Options B and C are also near-synonymous."),
]

# --------------------------------------------------------------- LOSSES (method wrong / 32B right)
LOSSES = [
("pmc-139",   "GENUINE",       "Histology panel C is pink/purple H&E; gold 'Hematoxylin and Eosin' matches the caption and is image-derivable."),
("pmc-343",   "GENUINE",       "Axial chest CT at heart level with a red arrow on an anterior right-sided ground-glass focus; 'middle lobe of the right lung' matches the caption and the level."),
("pmc-367",   "UNANSWERABLE",  "The question asks about an ECG, but the crop is a 2D echocardiogram with a red arrow. No ECG is shown."),
("pmc-2762",  "GENUINE",       "Angiographic panel (B) with 'Abnormal vessel' and 'Coronary sinus' labels in a lateral projection; gold 'Lateral view' matches the caption."),
("pmc-3085",  "BAD-GOLD",      "Crop is panel (f), a radiograph of the RIGHT SHOULDER/proximal humerus with an 'R' marker; the caption's tibial fracture is panel (g). For the image shown the 7B's 'X-ray of the upper extremities' is correct, so this 'loss' is a key artifact."),
("pmc-3258",  "UNANSWERABLE",  "Caption keys the answer to ASTERISKS, but the crop's markers are black arrowheads; the asterisk-marked follicles are not identifiable."),
("pmc-3315",  "GENUINE",       "Fetal brain MR; T1 vs T2 vs FLAIR weighting is a legitimate image call and gold 'T2-weighted' matches the caption."),
("pmc-4096",  "GENUINE",       "Axial iron-sensitive brain MR with a midbrain lesion; the plane is clearly axial and the caption's 'axial T2*' makes gold 'Axial T2-weighted' the only defensible option."),
("pmc-5238",  "GENUINE",       "CTP montage with green penumbra maps on the image-right of the brain (patient's left); gold 'Left MCA territory' matches the caption and the maps."),
("pmc-7203",  "UNANSWERABLE",  "AP lumbar radiograph with a radio-opaque deposit; 'local SOLID distribution pattern' vs option A's 'local FLUID distribution pattern' is a study-internal distinction available only from the caption."),
("pmc-7373",  "GENUINE",       "3D CTA reconstruction carries both markers the caption describes (arrowhead = common hepatic artery stenosis, arrow = splenic stump); the expert call is available in the image."),
("pmc-7387",  "UNANSWERABLE",  "The question asks about image E; the crop is panel C (the axial CT). The referenced panel is absent, and for the shown panel the 7B's 'Axial CT scan' is right."),
("pmc-8385",  "UNANSWERABLE",  "Which antibody was used (CD117 vs CD11b vs CD3 vs CD34) is not inferable from a stained section, and the crop is stained red while the caption keys the answer to CYAN areas."),
("pmc-8926",  "MULTI-CORRECT", "Caption says the blurring is of the 'gray-WHITE matter junction', so both option A (gray matter, gold) and option B (white matter, the 7B's answer) are defensible."),
("pmc-10439", "GENUINE",       "Selective bronchial angiogram with a black dotted circle over the patient's right lower zone; gold 'Right lower lobe' matches the caption and the marker."),
("pmc-11152", "UNANSWERABLE",  "Crop is a colour fundus photograph; the ELM is an OCT layer and the caption's referent is the OCT panel (E). The keyed finding cannot be seen on a fundus photo."),
("pmc-11290", "UNANSWERABLE",  "Question asks about the cervical cord MRI; the crop is panel (c), an axial BRAIN FLAIR. The cervical panel is absent."),
("pmc-11738", "UNANSWERABLE",  "Question asks what the thoracic CT revealed; the crop is panel (d), a T2-weighted axial MR of the axilla."),
("pmc-12324", "UNANSWERABLE",  "The crop shows green, red and yellow overlays; which of them is the MANUAL contour (vs the 3D-model intersection contour) is a caption-only mapping."),
("pmc-14227", "UNANSWERABLE",  "The labels visible in the crop are 'Rods' and 'NS'; the keyed ONL/OPL labels belong to other panels."),
("pmc-15052", "UNANSWERABLE",  "The question asks about the IVUS image, but the crop is the CT-angiography curved reformat (panel A). The IVUS panel is absent."),
("pmc-15473", "GENUINE",       "H&E panel C shows a nest of squamoid/basaloid tumour cells; 'Squamous cell carcinoma' matches the caption and is an expert-determinable call."),
("pmc-16248", "BAD-GOLD",      "Ill-posed: an anatomical structure is not 'located in a plane'. The caption means the cricoid was MEASURED in the transverse plane, while the crop is a LATERAL (sagittal) cervical radiograph - making the 7B's 'Sagittal' defensible for the image shown."),
("pmc-16191", "UNANSWERABLE",  "Single hip radiograph with no view label; 'which view showed the crescent sign' needs the multi-view figure, and FL/DL/LL are study-specific acronyms."),
("pmc-18696", "UNANSWERABLE",  "Non-medical: a TEM of nanoparticles cannot reveal whether the sample was a plant extract, animal extract, mineral or synthetic compound."),
("pmc-19263", "GENUINE",       "Axial contrast abdominal CT with bright aorta; arterial-phase CECT is an expert-determinable call and matches the caption."),
("pmc-21194", "MULTI-CORRECT", "The caption's own modality is 'STEM-HAADF' - scanning TRANSMISSION electron microscopy - so both gold A (TEM) and the 7B's C (SEM) capture part of it. Non-medical."),
("pmc-22842", "UNANSWERABLE",  "Which molecule the GREEN channel represents (GFP vs cytoplasm vs ribosomes) is a caption-only colour mapping, not visible content."),
("pmc-23794", "BAD-GOLD",      "The crop is the axial fused PET-CT the caption describes; MRI is mentioned only as a comparison. Gold 'MRI' contradicts the shown modality and the 7B's 'CT' is at least as defensible."),
("pmc-24225", "GENUINE",       "Axial pelvic CT shows a cecum-like structure left of midline; gold 'left lower abdomen' matches the caption and the finding is in the image."),
("pmc-24470", "UNANSWERABLE",  "No coloured arrows are present anywhere in the crop; the question is keyed to the YELLOW arrow specifically."),
("pmc-25101", "UNANSWERABLE",  "Non-medical (rape seed). Hypocotyl vs cotyledon vs epicotyl vs stem is not determinable from a 3D cell-wall rendering."),
("pmc-25236", "BAD-GOLD",      "Caption: the subcortical segmentation is 'overlaid on a CORONAL SLICE of the raw data', i.e. option C. Gold A ('overlaid on the cortical structures') contradicts the caption, and the crop (panel B) shows no coronal slice."),
("pmc-26113", "GENUINE",       "Whole-body bone scan shows increased uptake along the long bones of BOTH upper and lower limbs; gold 'periostosis ... upper and lower extremities' matches and the distribution is visible."),
("pmc-26955", "UNANSWERABLE",  "The crop is an axial chest CT showing thickened pericardium; no salivary gland is in the field, so parotid atrophy cannot be assessed."),
("pmc-27200", "UNANSWERABLE",  "The question asks about the contrast-enhanced CT; the crop is the coronal fused PET panel (which the caption describes as 'massive' FDG uptake). Ascites grading is also subjective."),
("pmc-28100", "GENUINE",       "Axial DWI shows a yellow arrow on a hyperintense lesion in the anterior part of the image-left (patient's right) breast; gold D matches."),
("pmc-28639", "GENUINE",       "Motion-degraded axial brain MR of an immature brain; newborn vs child is a hard but legitimate image call and gold matches the caption."),
("pmc-28701", "UNANSWERABLE",  "Question refers to the oblique image showing infundibular stenosis (caption panel A); the crop is panel B, a coronary-anatomy labelled image with no asterisk."),
("pmc-28829", "GENUINE",       "Fluoroscopic angiographic panel C shows catheters and a coil/basket device; 'Microcatheter insertion' is the only option consistent with an angiogram."),
("pmc-29217", "GENUINE",       "H&E ovary with black arrows on antral follicles; 'normal follicle' vs 'atretic follicle' is a standard histopathology call and only black arrows are present in the crop."),
("pmc-29954", "GENUINE",       "Sagittal cervical CT with a circle, arrows and a red asterisk on a collection POSTERIOR to the cord at C1-C2; gold 'between posterior dura and posterior C1 arch' matches."),
("pmc-30368", "MULTI-CORRECT", "Mitral annular calcification is a recognised cause of BOTH mitral stenosis (gold) and mitral regurgitation (option C); neither the image nor the caption resolves which."),
("pmc-30391", "GENUINE",       "Macroscopic cut surface is predominantly white and glistening/translucent; gold matches the caption and the image."),
("pmc-31032", "GENUINE",       "Clinical photographs of feet with short toes (brachydactyly); 'small hands and feet' is the only defensible option (extra toe / long digits are excluded by inspection)."),
("pmc-31596", "UNANSWERABLE",  "Which molecule the RED channel represents (ALP vs actin) is a caption-only colour mapping; the four-panel crop shows red/green/blue/merged with no legend."),
("pmc-31768", "GENUINE",       "3D CT of the occiput with the arrow on the incompletely ossified supraoccipital bone; vertical extension of the foramen magnum is the caption's finding and is an image feature."),
("pmc-31774", "UNANSWERABLE",  "'Appearance' vs 'DISAPPEARANCE' of a filling defect is a temporal statement requiring the prior study; a single CTA slice cannot support it."),
("pmc-32409", "BAD-GOLD",      "Malformed item: options A ('Coronal plane') and D ('Frontal plane') name the SAME plane, planes are not 'imaging techniques', and the crop is panel B (a dorsal/coronal-looking reformat) while the caption keys the sagittal panel C. The 7B's 'coronal' is defensible."),
("pmc-32480", "UNANSWERABLE",  "Pure publication-metadata question ('Where was the image originally published?'); nothing in the image bears on it, and 'Elsevier' is a publisher among three journals."),
]

# -------------------------------------------------- CONTROL (both models agree AND both marked correct)
CONTROL = [
("pmc-111",   "UNANSWERABLE",  "Laterality ('right side') cannot be read off a SAGITTAL MR, and the red box in the crop marks the pterygoid muscles rather than the subcutaneous component."),
("pmc-556",   "UNANSWERABLE",  "'Are the images to scale?' is figure metadata stated in the caption ('Images not to scale'); the dental-cast photograph carries no scale information."),
("pmc-2706",  "BAD-GOLD",      "The crop is a coronal head CT (bright bone, air-filled mastoids), not an MRI, and 'T1 diffusion-weighted' is not a real sequence. The caption itself is incoherent, and both models simply reproduced the caption-derived key."),
("pmc-3980",  "UNANSWERABLE",  "'Inserting' vs 'removing' is temporal and 'second branch' is a study-internal ordinal; neither is recoverable from two static angiographic frames."),
("pmc-4312",  "GENUINE",       "Enlarged periventricular panel with blue arrows shows a fan-shaped radial hyperintensity along deep veins; gold 'Radial' matches and is visible."),
("pmc-5842",  "GENUINE",       "Axial head CT in a bone window; gold 'CT scan' matches and the alternatives are excludable."),
("pmc-6006",  "GENUINE",       "Red rectangles enclose bright specks in a dark tomographic cross-section; 'Inclusions' is the only sensible option ('exclusions' is not a thing). Non-medical."),
("pmc-6045",  "GENUINE",       "Radiograph of the arm with an intramedullary nail in the humerus; gold 'Humerus' matches."),
("pmc-6567",  "UNANSWERABLE",  "A two-image comparison question, but only one radiograph (panel a) is in the crop."),
("pmc-6911",  "UNANSWERABLE",  "The crop is a tiny near-black panel with one red arrowhead; the left internal iliac / obturator fossa regions are not identifiable. Both models scored 'correct' on prior alone."),
("pmc-7322",  "GENUINE",       "Histology shows crypt-bearing mucosa without villi (colon) with focal denudation; 'Large intestine' is an expert-determinable call matching the caption."),
("pmc-8668",  "GENUINE",       "Coronal CT pulmonary angiogram with a red arrow on a filling defect; gold 'Computed Tomography' matches."),
("pmc-9050",  "UNANSWERABLE",  "The question asks about the T2WI appearance, but the crop is the H&E pathology panel (c) the caption describes."),
("pmc-11169", "GENUINE",       "Pelvic MR shows a large cystic mass with the acetabula/femoral heads in the field; gold 'Pelvic' matches."),
("pmc-11516", "GENUINE",       "Colour-Doppler echo with a labelled jet direction and a mosaic signal in the ventricular chamber; gold 'Ventricles' matches the caption."),
("pmc-13416", "MULTI-CORRECT", "Caption states the roof AND the anterior/posterior walls are intact, so option D (posterior wall) is as correct as gold A (roof)."),
("pmc-13448", "GENUINE",       "Two axial diffusion-weighted brain images with arrows; gold 'MRI' matches."),
("pmc-14111", "GENUINE",       "Coronal-oblique contrast CT of a gastric volvulus with the antrum labelled 'A' toward the patient's left; gold 'Left upper quadrant' matches the caption."),
("pmc-16002", "GENUINE",       "3D CT of the hemipelvis with plates/screws rendered and the joint reduced; 'satisfactory reduction and fixation' matches the caption and the alternatives (dislocation, unrelated fracture) are excludable."),
("pmc-17924", "GENUINE",       "Axial CT shows a giant cystic mass whose solid component lies on the image-left (patient's right); gold 'right quadrant' matches."),
("pmc-18032", "GENUINE",       "Coronal contrast abdominal CT with a striated left nephrogram; gold 'CT scan' matches."),
("pmc-18335", "GENUINE",       "Sagittal ankle MR shows a visibly multiloculated lesion at the distal tibia; gold 'Multicystic lesion' is the only descriptive option and matches the caption."),
("pmc-19364", "UNANSWERABLE",  "The question asks about the axial image OF THE THORAX; the crop is an abdominal slice (kidneys, para-aortic nodes). The paratracheal half of the keyed answer is not in the field."),
("pmc-19377", "GENUINE",       "The crop is an en-face (coronal) OCT-angiography MIP of a vascular plexus, plainly not a cross-sectional B-scan; gold 'Coronal' matches the caption."),
("pmc-19879", "GENUINE",       "H&E of jejunum with tall villi and labelled layers; gold 'Intestine' matches."),
("pmc-20043", "GENUINE",       "Sagittal lumbar CT with arrowheads; gold 'Sagittal' matches and is obvious."),
("pmc-20828", "GENUINE",       "B-mode/contrast ultrasound of the liver with a colour bar; gold 'Ultrasound' matches the caption's CEUS."),
("pmc-21440", "GENUINE",       "3D surface-rendered pelvic CT shows the acetabular fracture with a displaced femoral head; 'posterior wall fractures' matches the caption and is an expert-determinable call."),
("pmc-21710", "GENUINE",       "Caption-referential wording, but the crop is plainly an axial breast MRI with a segmented lesion; gold 'MRI' matches."),
("pmc-21711", "GENUINE",       "Light micrograph of a budding yeast cell with a 5 um bar; the two electron-microscopy options are excluded by inspection and gold 'Confocal Microscopy' matches the caption. Non-medical."),
("pmc-22623", "GENUINE",       "AP hip radiograph with joint-space narrowing and sclerosis and no prosthesis; gold 'Osteoarthritis of the hip' matches the pre-operative panel."),
("pmc-23875", "GENUINE",       "2D transoesophageal echo sector with colour Doppler and LA/RA labels; gold 'Two-dimensional (2D) imaging' matches."),
("pmc-24365", "UNANSWERABLE",  "The crop is a grayscale OCT texture image with no animal in the field; 'Mouse' comes from the setup schematic described in the caption."),
("pmc-24839", "UNANSWERABLE",  "Non-medical (shell microstructure). 1-degree vs 2/3/4-degree lamellae is a domain convention supplied by the caption."),
("pmc-24931", "MULTI-CORRECT", "The caption assigns plain white arrows to fusiform cells, dashed white arrows to bacteria and fat white arrows to protrusions - all of them white. Gold 'All of the above' and option A ('Fusiform cells') are both defensible."),
("pmc-25338", "GENUINE",       "The crop is a low-contrast plain radiograph of a pelvic tumour; gold 'X-ray' matches the caption and the alternatives are excludable."),
("pmc-26565", "GENUINE",       "Cardiac MR two-chamber view: the thick arrow is on bright pericardium over the anterior wall while the star marks the LV cavity, so the markers disambiguate; gold 'Left ventricular anterior wall' matches."),
("pmc-26736", "GENUINE",       "Featureless grey surface micrograph with a 10 um bar - a classic SEM dentin surface; gold 'Scanning electron microscopy' matches."),
("pmc-26993", "GENUINE",       "Confocal panel shows green FIBRE-like signal (not a circumscribed injection site) among magenta cell bodies; gold 'Projection fibers' is derivable from the morphology and matches the caption."),
("pmc-27089", "GENUINE",       "OCT-angiography slab in which the large superficial vessels are reproduced onto the deeper layer - the textbook projection artifact; gold C matches the caption."),
("pmc-27568", "GENUINE",       "Coronal chest CT centred on the sternoclavicular (medial clavicular) joints; gold matches and the scapular options are excluded by the field of view."),
("pmc-28489", "GENUINE",       "Coronal contrast abdominal CT with a rim-enhancing mass; gold 'CT' matches."),
("pmc-29051", "GENUINE",       "Shear-wave elastography overlay on B-mode; 'electrical stiffness' options are nonsense and blue = low stiffness is both the caption's statement and the universal SWE convention."),
("pmc-29264", "GENUINE",       "Full-length weight-bearing lower-extremity alignment radiograph with a ruler - by construction a standing film; gold 'Standing position' matches the caption."),
("pmc-30760", "GENUINE",       "Coronal brain MR with bright CSF in the ventricles (T2) and a normal hypothalamus; gold 'T2-weighted image with normal HT' matches."),
("pmc-31135", "GENUINE",       "Sagittal T2 with a yellow arrow on a dark (hypointense) proximal-tibial lesion; gold 'hypointense' matches."),
("pmc-31660", "GENUINE",       "Axial T1 brain MR of a young child at a level containing frontal and temporal cortex; the cerebellum/brainstem/occipital options are excluded by the level and gold matches the caption."),
("pmc-31795", "UNANSWERABLE",  "The crop is an axial head CT showing a haemorrhagic lesion; the venous-sinus study needed to say WHICH sinuses were COMPLETELY (vs partially) obliterated is absent."),
("pmc-32661", "GENUINE",       "MicroCT cross-section packed with trabeculae inside a cortical ring - a metaphysis, not a hollow diaphysis; gold 'Metaphysis' matches."),
("pmc-32713", "UNANSWERABLE",  "Pure caption-legend question ('What does the abbreviation H represent?'); the letter H is not even rendered in the crop."),
]

if __name__ == "__main__":
    out = {}
    for group in (WINS, LOSSES, CONTROL):
        for iid, cls, reason in group:
            key = iid if iid not in out else iid  # ids are unique within a group
            out.setdefault(key, dict(**{"class": cls, "reason": reason}))
    # wins/losses/control never share ids in this draw; assert that
    ids = [i for g in (WINS, LOSSES, CONTROL) for i, _, _ in g]
    assert len(ids) == len(set(ids)), "duplicate item id across groups"
    assert len(WINS) == 100 and len(LOSSES) == 50 and len(CONTROL) == 50, (len(WINS), len(LOSSES), len(CONTROL))
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(out)} classifications -> {OUT}")
    from collections import Counter
    for name, g in (("wins", WINS), ("losses", LOSSES), ("control", CONTROL)):
        print(name, len(g), dict(Counter(c for _, c, _ in g)))
