import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def match_keywords(user_prompt):
    # Define keyword lists for each disorder
    depression_keywords = [
    "sadness","sad","hopeless", "hopelessness","helpless", "helplessness", "despair", "guilt","worthless", "worthlessness",
    "isolation", "fatigue", "insomnia", "appetite changes", "suicidal thoughts",
    "withdrawal", "irritability", "irritable","lack of interest", "loss of pleasure", "low energy",
    "physical aches and pains","negative thoughts", "indecisiveness", "feeling overwhelmed", "overwhelming"
    "difficulty concentrating", "diminished ability to think",

    # Behavioral changes
    "changes in appetite", "sleep disturbances", "anger", "irritability",
    "hypervigilance", "psychomotor agitation", "psychomotor retardation",
    "increased use of alcohol or drugs", "reckless behavior", "social isolation",
    "changes in personal hygiene",

    # Emotional changes
    "feelings of worthlessness", "feeling disconnected from others",
    "feelings of emptiness", "anhedonia",

    # Social changes
     "family conflicts", "financial stress"
    ]

# Words associated with Anxiety
    anxiety_keywords = [
    "worry", "worried", "worrying" ,"fear", "nervousness","nervous", "panic","panicking", "restlessness","restless" "rapid heartbeat",
    "sweating", "trembling", "shortness of breath", "chest tightness", "dizziness",
    "feeling of impending doom", "avoidance behavior", "obsessive thoughts", "obsession"
    "compulsive behaviors", "muscle tension", "difficulty relaxing", "irritability",
    "trouble sleeping", "digestive issues","difficulty concentrating", "racing thoughts", "anticipatory anxiety",
    "catastrophizing", "overthinking",

    # Physical sensations
    "dry mouth", "fatigue", "headache", "nausea", "frequent urination",

    # Behavioral changes
    "social withdrawal", "increased substance use", "procrastination",

    # Emotional changes
    "feeling on edge", "feeling overwhelmed", "impatience", "anger outbursts",
    "tearfulness",

    # Sleep problems
    "difficulty falling asleep", "waking up during the night", "nightmares",

    # Social anxiety
    "fear of public speaking", "fear of social situations", "feeling judged or scrutinized",

    # Generalized anxiety
    "excessive worry", "feeling constantly on edge"
    ]

# Words associated with ADHD
    adhd_keywords = [
    "inattention", "hyperactivity","hyperactive", "impulsivity","impulsive", "distracted", "forgetfulness","forgetful",
    "disorganization","unorganised", "restlessness", "restless","difficulty focusing", "difficulty following instructions",
    "fidgeting", "interrupting", "difficulty waiting for turn", "risk-taking behavior",
    "time management issues", "poor planning", "procrastination", "difficulty completing tasks",
    "impatience","impatience", "recklessness","reckless", "mood swings",   "emotional dysregulation", "low self-esteem", "rejection sensitivity",

    # Social Difficulties
    "difficulty maintaining friendships", "trouble taking turns", "social awkwardness",

    # Executive Functioning Issues
    "working memory problems", "time blindness", "poor planning and organization", 
    "difficulty starting tasks",

    # Learning Difficulties
    "difficulty following instructions", "careless mistakes", "difficulty with complex tasks",

    # Sensory Processing
    "sensory overload", "sensory seeking", "difficulty filtering out background noise",

    # Motivation and Apathy
    "difficulty getting started on tasks", "difficulty sustaining motivation", "hyperfixation",

    # Physical Symptoms
    "difficulty falling asleep", "staying asleep", "increased appetite", "changes in eating habits",
    "headaches", "stomachaches",

    # Other
    "creativity", "problem-solving skills", "boredom susceptibility", "time perception",
    "internal monologue"
]

# Words associated with Eating Disorders
    eating_disorder_keywords = [
    "restriction", "bingeing", "purging", "obsession with food", "body dissatisfaction",
    "distorted body image", "fear of weight gain", "preoccupation with weight", "shame",
    "guilt", "secrecy", "social withdrawal", "low self-esteem", "anxiety around food",
    "excessive exercise", "food rituals", "denial of hunger", "physical health problems",
    "dental issues", "osteoporosis",  "perfectionism", "comparison", "dietary rigidity", "skipping meals", "stockpiling food",
    "lying about eating habits", "binge food preferences","overweight","malnourished"

  # Emotional Changes
    "isolation", "depression", "irritability", "anxiety",

  # Physical Symptoms
    "fatigue", "hair loss", "skin problems", "digestive issues", "irregular periods",
    "sleep disturbances",

  # Other
    "electrolyte imbalance", "delayed growth and development", "suicidal thoughts"
    ]

# Words associated with PTSD
    ptsd_keywords = [
    "flashbacks", "nightmares", "intrusive memories", "avoidance","avoiding", "emotional numbness","numb"
    "hypervigilance", "startle response", "irritability","irritable", "anger outbursts", "angry","guilt","gulity",
    "shame", "ashamed","feelings of detachment", "detached","difficulty concentrating", "sleep disturbances",
    "self-destructive behavior", "substance abuse", "depression", "anxiety",
    "relationship problems", "headaches", "stomachaches", "feeling zoned out", "feeling zoned in", "feeling on edge", "short temper",
    "feeling shut down", "blaming oneself", "feeling messed up",

  # Physical
    "stomach flips", "body aches", "feeling jittery", "trouble sleeping",
    "feeling tired all the time", "chest tightness",

  # Behavioral
    "avoiding triggers", "isolating oneself", "anger issues", "acting reckless","angry"
    "changes in appetite", "school problems",

  # Social
    "feeling disconnected", "arguments with friends", "feeling misunderstood",
    "trouble at home",

  # Informal Language
    "feeling out of it", "feeling fried", "head messed up", "can't handle it",
    "freaking out"
    ]

    # Tokenize user prompt and remove stopwords
    tokens = word_tokenize(user_prompt.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Initialize lists to store disorder-related keywords
    depression_matches = []
    anxiety_matches = []
    ptsd_matches = []
    adhd_matches = []
    eating_disorder_matches = []

    # Match keywords in the user prompt
    for word in filtered_tokens:
        if word in depression_keywords:
            depression_matches.append(word)
        if word in anxiety_keywords:
            anxiety_matches.append(word)
        if word in ptsd_keywords:
            ptsd_matches.append(word)
        if word in adhd_keywords:
            adhd_matches.append(word)
        if word in eating_disorder_keywords:
            eating_disorder_matches.append(word)

    # Print the names of lists with matches
    lists_with_matches = []
    if depression_matches:
        lists_with_matches.append("Depression")
    if anxiety_matches:
        lists_with_matches.append("Anxiety")
    if ptsd_matches:
        lists_with_matches.append("PTSD")
    if adhd_matches:
        lists_with_matches.append("ADHD")
    if eating_disorder_matches:
        lists_with_matches.append("Eating disorder")

    if lists_with_matches:
        return lists_with_matches[0]
    else:
        # Return a default value or handle the situation accordingly
        return "No disorder keywords found in the user prompt"


