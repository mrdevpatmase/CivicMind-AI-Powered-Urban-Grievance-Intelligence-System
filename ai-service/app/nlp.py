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
    "Big pothole on road making it dangerous to drive",
    "Garbage not collected from our area since 3 days",
    "Streetlight not working near my house",
    "Dirty water coming from tap",
    "Drainage overflowing and causing smell",

    # Road issues
    "Large crack in the middle of the road causing accidents",
    "Road surface completely broken after recent rain",
    "Deep potholes near the bus stop creating traffic problems",
    "Uneven road making it difficult for two-wheelers",
    "Road repair work left incomplete for weeks",
    "Waterlogging on damaged road blocking vehicles",
    "Sharp stones on road damaging vehicle tyres",
    "Collapsed road edge near drainage line",
    "Main road full of holes and unsafe at night",
    "Road sinking near construction site",
    "Broken speed breaker creating risk for drivers",
    "Road patchwork damaged again within days",

    # Waste issues
    "Garbage pile growing daily near residential area",
    "Dustbin overflowing and spreading bad smell",
    "No garbage truck visiting our street regularly",
    "Plastic waste scattered all over the road",
    "Unclean surroundings due to uncollected trash",
    "Garbage burning causing air pollution",
    "Public dustbin missing from the colony",
    "Waste dumped near school entrance",
    "Rotten food waste attracting animals",
    "Street corner filled with garbage bags",
    "Cleaning staff not coming for many days",
    "Garbage blocking roadside drainage",

    # Streetlight issues
    "Streetlight flickering continuously at night",
    "Entire street dark due to nonfunctional lights",
    "Broken streetlight pole creating danger",
    "Lights turning off frequently after sunset",
    "Dim streetlight reducing visibility on road",
    "No lighting near park area at night",
    "Streetlight damaged during storm not repaired",
    "Loose electric wires near streetlight pole",
    "Multiple lights not working in same lane",
    "Timer issue causing late streetlight activation",
    "Streetlight glass broken and unsafe",
    "Dark road increasing theft risk",

    # Water issues
    "Low water pressure in morning supply",
    "Water supply stopped without notice",
    "Mud mixed in tap water making it unusable",
    "Leakage in main water pipeline wasting water",
    "Irregular water timing affecting residents",
    "Contaminated drinking water causing illness",
    "No water supply for last two days",
    "Water tank overflow not controlled",
    "Pipeline burst flooding nearby houses",
    "Salty taste in supplied drinking water",
    "Water meter leakage near connection",
    "Very slow water flow in taps",

    # Sanitation / drainage issues
    "Sewage water flowing on the street",
    "Blocked drainage causing mosquito breeding",
    "Open manhole creating safety hazard",
    "Drain not cleaned for many months",
    "Bad smell coming from sewer line",
    "Overflowing gutter during rainfall",
    "Dirty surroundings due to poor sanitation",
    "Public toilet not cleaned regularly",
    "Drain cover broken and dangerous",
    "Standing dirty water near houses",
    "Sewage leakage near school boundary",
    "Unhygienic conditions causing health problems"
]


for s in samples:
    print(s, "→", predict_category(s))