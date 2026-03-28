import os
import json
import random
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

with open("./jab/data/user_data.json", "r") as file:
    user_data = json.load(file)


def training(data):
    """Build training intent list from user profile data."""

    def skill_intent(skill, exp):
        return {
            "patterns": [
                f"How many years of experience do you have in {skill}?",
                f"Experience in {skill}",
                f"How long have you been working with {skill}?",
                f"Years of {skill} experience?",
                f"Rate your {skill} expertise.",
                f"Do you know {skill}? How long?",
            ],
            "tag": skill + "_exp",
            "answer": str(exp),
        }

    training_data = [
        {
            "patterns": [
                "What is your full name?",
                "Tell me your full name.",
                "What is your name?",
                "Can I have your full name?",
                "Please provide your complete name.",
                "Your name?",
            ],
            "tag": "full_name",
            "answer": data["Full name"],
        },
        {
            "patterns": [
                "What is your first name?",
                "Tell me your first name.",
                "What is your given name?",
                "Can I know your first name?",
                "First name please.",
            ],
            "tag": "first_name",
            "answer": data["First name"],
        },
        {
            "patterns": [
                "What is your last name?",
                "What is your surname?",
                "Tell me your last name.",
                "Your family name?",
                "Last name?",
            ],
            "tag": "last_name",
            "answer": data["Last name"],
        },
        {
            "patterns": [
                "What is your gender?",
                "What do you identify as?",
                "Gender",
                "Please select your gender.",
                "Male or Female?",
            ],
            "tag": "gender",
            "answer": data["Gender"],
        },
        {
            "patterns": [
                "What is your mobile number?",
                "Tell me your phone number.",
                "What is your cell phone number?",
                "Can I have your mobile phone number?",
                "Contact number?",
                "Phone?",
            ],
            "tag": "mobile",
            "answer": data["Mobile"],
        },
        {
            "patterns": [
                "What is your email address?",
                "Tell me your email.",
                "What is your email?",
                "Can I have your email?",
                "Email address please.",
                "Your email ID?",
            ],
            "tag": "email",
            "answer": data["Email"],
        },
        {
            "patterns": [
                "Where are you located?",
                "What is your location?",
                "Where do you live?",
                "Your current city?",
                "Which state are you from?",
                "Current location?",
                "Which city do you reside in?",
            ],
            "tag": "location",
            "answer": data["Location"],
        },
        {
            "patterns": [
                "What is your current salary?",
                "What is your CTC?",
                "How much are you earning?",
                "What is your income?",
                "Current annual package?",
                "Present salary?",
            ],
            "tag": "current_salary",
            "answer": data["Salary"],
        },
        {
            "patterns": [
                "What is your date of birth?",
                "When is your birthday?",
                "What is your DOB?",
                "Your birth date?",
                "Date of birth?",
            ],
            "tag": "date_of_birth",
            "answer": data["Date of birth"],
        },
        {
            "patterns": [
                "What is your expected salary?",
                "What is your desired CTC?",
                "What salary are you expecting?",
                "Your expected income?",
                "Expected package?",
                "What is your salary expectation?",
            ],
            "tag": "expected_salary",
            "answer": data["Expected salary"],
        },
        {
            "patterns": [
                "What is your preferred work location?",
                "Where do you prefer to work?",
                "What is your desired job location?",
                "Your preferred city for work?",
                "Preferred location?",
            ],
            "tag": "preferred_location",
            "answer": data["Preferred work location"],
        },
        {
            "patterns": [
                "What is your notice period?",
                "When can you join?",
                "What is your availability to join?",
                "How soon can you start?",
                "Joining time?",
                "How many days notice period do you have?",
                "Earliest joining date?",
            ],
            "tag": "notice_period",
            "answer": data["Notice period"],
        },
        {
            "patterns": [
                "How many years of experience do you have?",
                "What is your total experience?",
                "How much experience do you have?",
                "Your total work experience?",
                "Current professional experience?",
                "Total years in industry?",
                "Years of work experience?",
            ],
            "tag": "current_experience",
            "answer": data["Total experience"],
        },
        {
            "patterns": [
                "Have you worked at our organization?",
                "Were you previously employed by this organization?",
                "Did you work for us before?",
                "Any prior employment here?",
            ],
            "tag": "bool_employee",
            "answer": "No",
        },
        {
            "patterns": [
                "How should we address you?",
                "What is your salutation?",
                "What is your title?",
                "How do you prefer to be addressed?",
                "Mr., Ms., or Dr.?",
            ],
            "tag": "salutation",
            "answer": data["Salutation"],
        },
        {
            "patterns": [
                "I consent to the Privacy Policy",
                "Do you agree to the Privacy Policy?",
                "I accept the Privacy Policy",
                "Consent to the Privacy Policy",
                "Do you agree to our terms?",
                "Accept terms and conditions?",
            ],
            "tag": "privacy_policy_consent",
            "answer": "Yes",
        },
        {
            "patterns": [
                "Are you currently residing in location or willing to relocate?",
                "Do you live in location or are you open to relocating?",
                "Are you based in location or would you move?",
                "Are you willing to relocate?",
                "Open to relocation?",
                "Would you move to a different city for this role?",
            ],
            "tag": "relocation_bool",
            "answer": "Yes",
        },
        {
            "patterns": [
                "Are you currently employed?",
                "What is your employment status?",
                "Are you working currently?",
                "Current job status?",
            ],
            "tag": "employment_status",
            "answer": "Yes",
        },
        {
            "patterns": [
                "What is your highest qualification?",
                "Your educational background?",
                "Highest degree obtained?",
                "What did you study?",
                "Your academic qualification?",
            ],
            "tag": "education",
            "answer": data.get("Highest qualification", "Bachelor's Degree"),
        },
        {
            "patterns": [
                "Are you a fresher or experienced?",
                "How would you describe yourself?",
                "Fresher or experienced professional?",
            ],
            "tag": "fresher_or_exp",
            "answer": "Experienced" if data.get("Total experience", "0") not in ("0", "") else "Fresher",
        },
        {
            "patterns": [
                "Do you have a passport?",
                "Do you hold a valid passport?",
                "Passport available?",
            ],
            "tag": "passport",
            "answer": data.get("Passport", "Yes"),
        },
        {
            "patterns": [
                "Are you willing to work in shifts?",
                "Can you work in rotational shifts?",
                "Night shift availability?",
                "Shift preference?",
            ],
            "tag": "shift_willingness",
            "answer": data.get("Shift willingness", "Yes"),
        },
        {
            "patterns": [
                "Do you have a two-wheeler?",
                "Do you own a vehicle?",
                "Do you have your own vehicle?",
                "Own a bike or car?",
            ],
            "tag": "vehicle",
            "answer": data.get("Vehicle", "Yes"),
        },
    ]

    # Dynamic skill intents
    if data.get("Skills"):
        for skill, exp in data["Skills"].items():
            training_data.append(skill_intent(skill, exp))
        training_data.append(
            {
                "patterns": [
                    "What are your key skills?",
                    "Tell me your skills.",
                    "What skills do you have?",
                    "Your key competencies?",
                    "List your technical skills.",
                    "What technologies do you know?",
                ],
                "tag": "Skills",
                "answer": ", ".join(str(k) for k in data["Skills"].keys()),
            }
        )

    return training_data


class ChatbotBuild:
    def __init__(self, username):
        self.username = username
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = ['?', '!', '.', ',']
        self.user_data = user_data
        self.patterns = []
        self.tags = []
        self.responses = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.model = None
        self.training_data = training(self.user_data)
        self._load_data()
        self._preprocess()
        self._build_model()

    def _load_data(self):
        for intent in self.training_data:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.patterns.append((tokens, intent["tag"]))
                if intent["tag"] not in self.tags:
                    self.tags.append(intent["tag"])
            self.responses[intent["tag"]] = intent["answer"]

    def _preprocess(self):
        for tokens, tag in self.patterns:
            for word in tokens:
                if word not in self.ignore_words:
                    self.words.append(self.lemmatizer.lemmatize(word.lower()))
            self.documents.append((tokens, tag))
            if tag not in self.classes:
                self.classes.append(tag)
        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))

        training_rows = []
        empty_output = [0] * len(self.classes)
        for tokens, tag in self.documents:
            bow = [1 if w in [self.lemmatizer.lemmatize(t.lower()) for t in tokens] else 0 for w in self.words]
            row_output = list(empty_output)
            row_output[self.classes.index(tag)] = 1
            training_rows.append([bow, row_output])
        random.shuffle(training_rows)
        data = np.array(training_rows, dtype=object)
        self.train_x = list(data[:, 0])
        self.train_y = list(data[:, 1])

    def _build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(len(self.train_x[0]),)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.classes), activation='softmax'),
        ])
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy'],
        )

    def train_model(self, epochs=150):
        self.model.fit(
            np.array(self.train_x), np.array(self.train_y),
            epochs=epochs, batch_size=5, verbose=1
        )
        out_dir = f"./jab/data/{self.username}"
        os.makedirs(out_dir, exist_ok=True)
        self.model.save(f"{out_dir}/model.keras")
        print("Model saved to", f"{out_dir}/model.keras")
