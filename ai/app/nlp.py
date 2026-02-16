import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

model = joblib.load(os.path.join(MODEL_DIR, "nlp_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

def predict_category(text: str):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = float(max(model.predict_proba(vec)[0]))
    return pred, round(prob, 2)



samples = [

# --- Road ---
"Road full of cracks near market area",
"Newly built road already damaged",
"Broken road causing daily traffic jam",
"Potholes filled with dirty water",
"Road shoulder collapsed after rain",
"Loose gravel on main road causing skidding",
"Heavy bumps making driving unsafe",
"Road divider broken in middle",
"Damaged road near school zone",
"Construction debris left on road",
"Road repair pending for months",
"Vehicles stuck due to bad road condition",
"Road sinking slowly near bridge",
"Side road completely unusable",
"Main junction road badly damaged",
"Sharp turn road surface broken",
"Road dug and not repaired",
"Uneven patchwork creating accidents",
"Damaged road slowing ambulance movement",
"Highway service lane full of potholes",

# --- Waste ---
"Garbage overflowing near bus stand",
"Trash not cleared from colony street",
"Rotting garbage causing foul smell",
"Municipal bin always full",
"Garbage scattered by animals",
"No cleaning in public area",
"Waste dumped near temple",
"Garbage pile attracting mosquitoes",
"Plastic waste blocking footpath",
"Overflowing community dustbin",
"Garbage lying near hospital",
"Street sweeping not done regularly",
"Food waste thrown on roadside",
"Uncollected trash for one week",
"Garbage truck skipping our area",
"Dirty public surroundings everywhere",
"Open dumping ground near houses",
"Garbage creating health issues",
"Trash burning causing smoke",
"Unclean street due to waste",

# --- Streetlight ---
"Streetlight completely off at night",
"Pole light not repaired for weeks",
"Area dark after sunset",
"Streetlight blinking continuously",
"Electric light wire hanging loose",
"Multiple lights fused in lane",
"No lighting near main road",
"Streetlight damaged by storm",
"Dim lights causing visibility issue",
"Streetlight timer not working",
"Broken pole leaning dangerously",
"Lights off near public park",
"Dark street increasing crime fear",
"Streetlight glass shattered",
"Power supply issue in lights",
"Streetlight repair complaint ignored",
"Old lights not replaced",
"Frequent light failure at night",
"Lights not turning on automatically",
"Unsafe darkness in residential lane",

# --- Water ---
"No drinking water supply today",
"Water pressure extremely low",
"Tap water muddy and dirty",
"Pipeline leaking continuously",
"Water timing very irregular",
"Contaminated water smell present",
"No water for three days",
"Overflowing water tank wasting water",
"Pipe burst flooding road",
"Salty water coming in taps",
"Meter connection leaking",
"Water flow too slow",
"Supply stopped without notice",
"Rusty colored water from tap",
"Water shortage in morning hours",
"Broken pipeline not repaired",
"Dirty water causing sickness",
"Low pressure on upper floors",
"Water valve not functioning",
"Daily water interruption issue",

# --- Sanitation ---
"Drainage blocked near house",
"Sewage overflowing on street",
"Open manhole without cover",
"Drain not cleaned properly",
"Bad sewer smell in area",
"Gutter overflowing during rain",
"Public toilet extremely dirty",
"Dirty stagnant water nearby",
"Broken drain cover dangerous",
"Mosquito breeding in drain",
"Unhygienic street conditions",
"Sewage leakage spreading smell",
"Dirty drainage near school",
"Clogged sewer line issue",
"Drain water entering houses",
"Garbage mixed in drainage",
"Open sewage pit unsafe",
"Drain blockage causing flooding",
"Health risk due to sanitation",
"Overflowing sewer near road",

# --- Mixed realistic complaints ---
"Road and drainage both damaged in our lane",
"Garbage and sewage smell everywhere",
"No streetlight and unsafe road at night",
"Water leakage damaging nearby road",
"Blocked drain causing mosquito problem",
"Garbage pile near broken streetlight",
"Overflowing sewer next to school road",
"Waterlogging due to poor drainage",
"Dirty surroundings and waste dumping",
"Dark street with potholes everywhere",
"Streetlight off and garbage lying nearby",
"Pipeline burst creating road damage",
"Unclean public toilet near market",
"Drainage water mixing with road water",
"Garbage burning near residential houses",
"Broken road and no lighting",
"Dirty water supply in colony",
"Blocked sewer causing smell in street",
"Overflowing garbage near hospital",
"Unsafe open manhole in road",

# --- Extra natural phrasing ---
"Please repair the damaged road urgently",
"No one is collecting garbage here",
"Streetlight complaint already given but no action",
"Tap water quality is very poor",
"Drainage problem increasing daily",
"Road condition worst after rainfall",
"Garbage issue making area unhealthy",
"Lights not working since last month",
"Water supply timing not fixed",
"Sanitation workers not visiting area",
"Road repair work incomplete",
"Garbage smell unbearable",
"Street too dark at night",
"Water leakage wasting resources",
"Drain blockage causing insects",
"Unsafe walking due to potholes",
"Overflowing bins everywhere",
"Electric pole light not functioning",
"Dirty drinking water supply",
"Sewer line maintenance required",

# --- Final set ---
"Immediate action needed for road damage",
"Garbage cleaning required urgently",
"Streetlight maintenance pending",
"Water shortage complaint again",
"Drain cleaning not done",
"Road repair request submitted earlier",
"Waste removal vehicle not coming",
"Streetlight off near main gate",
"Water pipeline issue unresolved",
"Sanitation condition very poor",
"Broken road causing accidents daily",
"Garbage spreading diseases",
"Street dark and unsafe",
"Water supply too weak",
"Drain overflow during rain",
"Road potholes increasing",
"Trash everywhere in colony",
"Lights fused in entire lane",
"Dirty water affecting health",
"Sewer blockage urgent issue"
]



for s in samples:
    print(s, "→", predict_category(s))