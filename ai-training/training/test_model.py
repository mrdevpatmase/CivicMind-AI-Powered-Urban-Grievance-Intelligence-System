from ai.inference.nlp import predict_category

samples = [
    "Big pothole on road making it dangerous to drive",
    "Garbage not collected from our area since 3 days",
    "Streetlight not working near my house",
    "Dirty water coming from tap",
    "Drainage overflowing and causing smell"
]

for s in samples:
    print(s, "→", predict_category(s))
