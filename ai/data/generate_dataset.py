import csv
import random

OUTPUT_PATH = "complaints_large.csv"

categories = {
    "water": [
        "no water supply since morning",
        "pani kal se nahi aa raha",
        "dirty water coming from tap",
        "tap se brown pani aa raha",
        "water pressure very low",
        "pipeline leakage wasting water",
        "pani me smell aa rahi hai",
        "irregular water timing in colony",
    ],
    "road": [
        "road full of potholes",
        "sadak bahut kharab hai",
        "bike chalana mushkil due to broken road",
        "gadde bahut bade hai",
        "road repair work incomplete",
        "uneven road causing accidents",
        "waterlogging on damaged road",
        "new road already damaged",
    ],
    "waste": [
        "garbage not collected for days",
        "kachra har jagah pada hai",
        "dustbin overflow ho raha hai",
        "rotting garbage causing smell",
        "plastic waste everywhere",
        "cleaning staff not coming",
        "garbage truck skipping area",
        "public place very dirty",
    ],
    "streetlight": [
        "streetlight not working at night",
        "light band hai 3 din se",
        "area completely dark",
        "pole light blinking",
        "electric light fused",
        "streetlight broken after rain",
        "timer issue in streetlight",
        "no lighting near park",
    ],
    "sanitation": [
        "drain blocked and smelling",
        "naali jam ho gayi hai",
        "sewer water on road",
        "open manhole dangerous",
        "mosquito breeding in drain",
        "public toilet very dirty",
        "gutter overflow in rain",
        "dirty drainage near houses",
    ],
}

def generate_sentence(base):
    prefixes = ["please", "urgent", "sir", "complaint:", "issue:", ""]
    suffixes = ["jaldi fix kare", "need repair", "bahut problem ho rahi", "", "immediately required"]

    return f"{random.choice(prefixes)} {base} {random.choice(suffixes)}".strip()


rows = []

for category, samples in categories.items():
    for _ in range(240):  # 240 × 5 = 1200
        base = random.choice(samples)
        text = generate_sentence(base)
        rows.append([text, category])

random.shuffle(rows)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "category"])
    writer.writerows(rows)

print(f"✅ Dataset generated with {len(rows)} samples → {OUTPUT_PATH}")
