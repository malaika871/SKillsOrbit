import re
import io
import fitz  # PyMuPDF
import spacy
from rapidfuzz import fuzz
from docx import Document

SKILL_TAXONOMY = [
    # Programming Languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "R", "Go",
    "Rust", "Swift", "Kotlin", "PHP", "Ruby", "Scala", "MATLAB", "Perl",
    "Bash", "PowerShell", "Haskell", "Dart", "Lua",
    # Web / Frontend
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js",
    "Django", "Flask", "Spring", "Bootstrap", "jQuery", "Next.js", "Nuxt.js",
    "Redux", "GraphQL", "REST API", "WebSockets", "Webpack", "Sass",
    "Tailwind CSS", "FastAPI", "Laravel", "Rails",
    # Data Science / ML / AI
    "Machine Learning", "Deep Learning", "Neural Networks", "NLP",
    "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau", "Power BI",
    "Statistics", "Probability", "Data Visualization", "Feature Engineering",
    "Model Deployment", "Jupyter", "NLTK", "OpenCV", "XGBoost", "LightGBM",
    "Data Mining", "Time Series", "A/B Testing", "Spark", "Hadoop",
    "Airflow", "ETL", "Data Engineering", "MLOps",
    # Databases
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle",
    "Cassandra", "DynamoDB", "Elasticsearch", "Firebase", "Neo4j", "NoSQL",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins",
    "GitHub Actions", "Terraform", "Ansible", "Linux", "Git", "GitHub",
    "GitLab", "Nginx", "Heroku", "Vercel", "DevOps", "Microservices",
    # Security
    "Cybersecurity", "Network Security", "Penetration Testing", "OWASP",
    "Cryptography", "Firewalls", "SIEM", "Ethical Hacking",
    # Mobile
    "Android", "iOS", "React Native", "Flutter",
    # Architecture / Networking
    "TCP/IP", "HTTP/HTTPS", "Load Balancing", "API Gateway",
    "Kafka", "RabbitMQ", "Distributed Systems", "REST",
    # Tools
    "Excel", "Jira", "Confluence", "Postman", "Figma", "Adobe XD",
    # Soft Skills
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Project Management", "Time Management",
    "Agile", "Scrum", "Kanban",
    # Other
    "Blockchain", "IoT", "Unity", "Unreal Engine", "Embedded Systems",
    "Arduino", "Raspberry Pi", "Quantitative Analysis", "Financial Modeling",
    "SEO", "Digital Marketing", "Bioinformatics",
]

EDUCATION_KEYWORDS = {
    "PhD": ["phd", "ph.d", "doctorate", "doctoral"],
    "Master": ["master", "msc", "m.sc", "ms ", "m.s.", "mba", "m.eng", "m.tech"],
    "Bachelor": ["bachelor", "bsc", "b.sc", "bs ", "b.s.", "b.eng", "b.tech", "b.e."],
    "Associate": ["associate", "a.a.s"],
    "Diploma": ["diploma", "certificate", "certification"],
    "High School": ["high school", "secondary", "matric", "a-level", "o-level"],
}

_EMAIL_RE     = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
_PHONE_RE     = re.compile(r'(?<!\d)(?:\+?\d[\d\s\-().]{7,15}\d)(?!\d)')
# Name validation helpers
_NAME_ALPHA   = re.compile(r"^[A-Za-z][A-Za-z '\-]{1,50}$")          # letters/spaces/hyphens only
_NAME_SKIP    = re.compile(r"[/:]|\.com|\.org|\.net|\.io|http|www\.", re.I)  # URL fragments
_NAME_HDR     = re.compile(                                            # section headers to reject
    r"^(summary|experience|education|skills|projects|certifications|"
    r"profile|objective|contact|references|languages|awards|publications)$",
    re.I,
)

# Flat lowercase set of all taxonomy terms for O(1) look-up
_TAXONOMY_SET: set[str] = {s.lower() for s in (
    # keep in sync with SKILL_TAXONOMY — duplicated here so the set is
    # available before the class is instantiated
    "python","java","javascript","typescript","c","c++","c#","r","go","rust",
    "swift","kotlin","php","ruby","scala","matlab","perl","bash","powershell",
    "haskell","dart","lua","html","css","react","angular","vue.js","node.js",
    "express.js","django","flask","spring","bootstrap","jquery","next.js",
    "nuxt.js","redux","graphql","rest api","websockets","webpack","sass",
    "tailwind css","fastapi","laravel","rails","streamlit","machine learning",
    "deep learning","neural networks","nlp","computer vision","tensorflow",
    "pytorch","keras","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "tableau","power bi","statistics","probability","data visualization",
    "feature engineering","model deployment","jupyter","nltk","opencv",
    "xgboost","lightgbm","data mining","time series","a/b testing","spark",
    "hadoop","airflow","etl","data engineering","mlops","sql","mysql",
    "postgresql","mongodb","redis","sqlite","oracle","cassandra","dynamodb",
    "elasticsearch","firebase","neo4j","nosql","aws","azure","gcp","docker",
    "kubernetes","ci/cd","jenkins","github actions","terraform","ansible",
    "linux","git","github","gitlab","nginx","heroku","vercel","devops",
    "microservices","cybersecurity","network security","penetration testing",
    "owasp","cryptography","firewalls","siem","ethical hacking","android",
    "ios","react native","flutter","tcp/ip","http/https","load balancing",
    "api gateway","kafka","rabbitmq","distributed systems","rest","excel",
    "jira","confluence","postman","figma","adobe xd","communication",
    "leadership","teamwork","problem solving","critical thinking",
    "project management","time management","agile","scrum","kanban",
    "blockchain","iot","unity","unreal engine","embedded systems","arduino",
    "raspberry pi","quantitative analysis","financial modeling","seo",
    "digital marketing","bioinformatics",
)}


class SkillExtractor:
    def __init__(self):
        self._taxonomy_lower = [s.lower() for s in SKILL_TAXONOMY]
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "SpaCy model 'en_core_web_sm' not found. "
                "Please install it with: python -m spacy download en_core_web_sm"
            )

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return " ".join(page.get_text() for page in doc)
        except Exception:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return " ".join(page.extract_text() or "" for page in pdf.pages)

    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        doc = Document(io.BytesIO(file_bytes))
        return " ".join(para.text for para in doc.paragraphs)

    def extract_skills(self, text: str) -> list:
        text_lower = text.lower()
        found = set()

        # Exact word-boundary match
        for skill, skill_l in zip(SKILL_TAXONOMY, self._taxonomy_lower):
            if re.search(r'\b' + re.escape(skill_l) + r'\b', text_lower):
                found.add(skill)

        # Fuzzy match on tokens and bigrams not already matched
        tokens = re.findall(r'\b[\w.#+/]+\b', text_lower)
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        unmatched = [(s, sl) for s, sl in zip(SKILL_TAXONOMY, self._taxonomy_lower) if s not in found]

        for candidate in tokens + bigrams:
            for skill, skill_l in unmatched:
                if skill not in found and fuzz.ratio(candidate, skill_l) >= 85:
                    found.add(skill)

        return sorted(found)

    # ── name extraction helpers ──────────────────────────────────────────────

    @staticmethod
    def _is_valid_name(candidate: str) -> bool:
        """Return True if the string looks like a real person name."""
        c = candidate.strip()
        return (
            bool(_NAME_ALPHA.match(c))          # only letters / spaces / hyphens
            and not _NAME_SKIP.search(c)         # no URL fragments
            and not _NAME_HDR.match(c)           # not a section header
            and c.lower() not in _TAXONOMY_SET   # not a known skill/tool
        )

    @staticmethod
    def _extract_name_from_top(text: str) -> str:
        """
        Priority-1: scan the first 5 non-empty lines of the resume.
        A name line is typically 2-3 title-case words in the first 1-2 lines
        that are not an email, phone, URL, or taxonomy term.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()][:5]
        for line in lines:
            # Skip lines that are clearly contact info or section headers
            if _EMAIL_RE.search(line) or _PHONE_RE.search(line):
                continue
            if _NAME_SKIP.search(line):
                continue
            words = line.split()
            if not (2 <= len(words) <= 4):
                continue
            # Every word must start with an uppercase letter and be alphabetic
            if not all(w[0].isupper() and re.match(r"^[A-Za-z'\-]+$", w) for w in words):
                continue
            candidate = " ".join(words)
            if SkillExtractor._is_valid_name(candidate):
                return candidate
        return ""

    def _extract_name_from_ner(self, text: str) -> str:
        """
        Priority-2: use spaCy PERSON entities, but guard against tech-term
        false positives (Streamlit, Flask, etc.) and single-word matches.
        """
        doc = self.nlp(text[:2000])
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            candidate = ent.text.strip()
            words = candidate.split()
            if len(words) < 2:          # require at least first + last name
                continue
            if SkillExtractor._is_valid_name(candidate):
                return candidate
        return ""

    # ── public method ────────────────────────────────────────────────────────

    def extract_info(self, text: str) -> tuple:
        email_m = _EMAIL_RE.search(text)
        email   = email_m.group(0) if email_m else ""

        phone_m = _PHONE_RE.search(text)
        phone   = phone_m.group(0).strip() if phone_m else ""

        # Priority 1 → top-of-resume lines
        name = self._extract_name_from_top(text)
        # Priority 2 → spaCy NER with taxonomy + URL filtering
        if not name:
            name = self._extract_name_from_ner(text)
        # Priority 3 → graceful fallback
        if not name:
            name = "Name not detected"

        text_lower = text.lower()
        education  = ""
        for level, keywords in EDUCATION_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                education = level
                break

        return name, email, phone, education

    def analyze(self, file_bytes: bytes, filename: str) -> dict:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext == "pdf":
            text = self.extract_text_from_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            text = self.extract_text_from_docx(file_bytes)
        else:
            raise ValueError(
                f"Unsupported file type: .{ext}. Please upload a PDF or DOCX file."
            )

        name, email, phone, education = self.extract_info(text)
        skills = self.extract_skills(text)

        return {
            "name": name,
            "email": email,
            "phone": phone,
            "education": education,
            "skills_found": skills,
            "skills_count": len(skills),
            "raw_text_length": len(text),
            "_text": text,
        }


# ─── Petri Net ────────────────────────────────────────────────────────────────

_PLACES = ["P0", "P1", "P2", "P3", "P4", "P5"]

_PLACE_LABELS = {
    "P0": "Resume Uploaded",
    "P1": "Text Extracted",
    "P2": "Skills Identified",
    "P3": "Career Matched",
    "P4": "Gap Analysis Done",
    "P5": "Report Ready",
}

_TRANSITIONS = [
    {"id": "T0", "label": "extract_text",    "input": "P0", "output": "P1"},
    {"id": "T1", "label": "identify_skills", "input": "P1", "output": "P2"},
    {"id": "T2", "label": "match_careers",   "input": "P2", "output": "P3"},
    {"id": "T3", "label": "analyze_gaps",    "input": "P3", "output": "P4"},
    {"id": "T4", "label": "generate_report", "input": "P4", "output": "P5"},
]


class PetriNet:
    def __init__(self):
        self.places = _PLACES[:]
        self.transitions = _TRANSITIONS[:]
        self.tokens = {p: 0 for p in _PLACES}

    def add_token(self, place: str):
        self.tokens[place] += 1

    def is_enabled(self, transition: dict) -> bool:
        return self.tokens[transition["input"]] > 0

    def fire(self, transition: dict) -> bool:
        if not self.is_enabled(transition):
            return False
        self.tokens[transition["input"]] -= 1
        self.tokens[transition["output"]] += 1
        return True

    def simulate_all(self) -> dict:
        self.tokens = {p: 0 for p in _PLACES}
        self.add_token("P0")
        log = [
            {
                "step": 0,
                "event": "init",
                "description": "Token placed at P0: Resume Uploaded",
                "tokens": dict(self.tokens),
                "active_place": "P0",
                "fired_transition": None,
            }
        ]
        for t in self.transitions:
            if self.is_enabled(t):
                self.fire(t)
                log.append({
                    "step": len(log),
                    "event": "fire",
                    "transition": t["id"],
                    "label": t["label"],
                    "description": (
                        f"Fired {t['id']} ({t['label']}): "
                        f"token moved {t['input']} → {t['output']}"
                    ),
                    "tokens": dict(self.tokens),
                    "active_place": t["output"],
                    "fired_transition": t["id"],
                })
        return {
            "log": log,
            "final_tokens": dict(self.tokens),
            "places": _PLACES,
            "place_labels": _PLACE_LABELS,
            "transitions": _TRANSITIONS,
        }
