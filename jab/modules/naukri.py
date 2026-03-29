import time
import re
import os
import random
import json
import logging
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from playwright.sync_api import sync_playwright, expect, TimeoutError as PlaywrightTimeoutError
import tensorflow as tf
import nltk
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

# ─────────────────────────── helpers ────────────────────────────

def setup_logger(name, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def human_delay(min_ms=500, max_ms=2000):
    time.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


# ─────────────────────── ML chatbot model ───────────────────────

class ChatbotModel:
    def __init__(self, user_data):
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_words = ['?', '!', '.', ',']
        self.user_data = user_data
        self.words = []
        self.classes = []
        self._load_vocab()
        self._load_model()

    def _load_vocab(self):
        for intent in self.user_data:
            for pattern in intent['patterns']:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        self.words = sorted(set(
            self.lemmatizer.lemmatize(w.lower())
            for w in self.words if w not in self.ignore_words
        ))
        self.classes = sorted(set(self.classes))

    def _load_model(self):
        model_path = f"./jab/data/{_active_user}/model.keras"
        self.model = tf.keras.models.load_model(model_path)

    def _bow(self, sentence):
        tokens = [self.lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]
        bag = [1 if w in tokens else 0 for w in self.words]
        return np.array(bag)

    def predict_class(self, sentence):
        p = self._bow(sentence)
        res = self.model.predict(np.array([p]), verbose=0)[0]
        results = sorted(
            [[i, float(r)] for i, r in enumerate(res) if r > 0.25],
            key=lambda x: x[1], reverse=True
        )
        return [{"intent": self.classes[r[0]], "probability": r[1]} for r in results]

    def get_response(self, ints):
        if not ints:
            return 'A'
        tag = ints[0]['intent']
        for intent in self.user_data:
            if intent['tag'] == tag:
                return intent['answer']
        return 'A'

    def chatbot_response(self, msg):
        try:
            msg_lower = msg.lower()

            # Dynamic: last working day → compute 30 days from today
            last_day_keywords = ['last working day', 'last day', 'last date of employment', 'last date']
            if any(kw in msg_lower for kw in last_day_keywords):
                return (datetime.now() + timedelta(days=30)).strftime('%d/%m/%Y')

            response = self.get_response(self.predict_class(msg))

            if response == 'A':
                # Total / overall experience → 5 years
                total_exp_keywords = [
                    'total experience', 'total years', 'overall experience',
                    'years of experience', 'how many years', 'total work experience',
                ]
                if any(kw in msg_lower for kw in total_exp_keywords):
                    return '5'
                # Tool-specific / relevant / devops experience → 4 years
                specific_exp_keywords = [
                    'experience', 'years', 'how long', 'expertise',
                    'worked with', 'using', 'knowledge of', 'relevant',
                ]
                if any(kw in msg_lower for kw in specific_exp_keywords):
                    return '4'
            return response
        except Exception:
            return 'A'


# ─────────────────────── chatbot agent ──────────────────────────

class ChatbotAgent:
    def __init__(self, page, username, user_info=None, logger=None):
        global _active_user
        _active_user = username
        self.page = page
        self.user_info = user_info or {}
        self.logger = logger or setup_logger('ChatbotAgent')
        training_path = f"./jab/data/{username}/training_data.json"
        with open(training_path, 'r') as f:
            user_data = json.load(f)
        self.model = ChatbotModel(user_data)
        self.analyzer = SentimentIntensityAnalyzer()

    # ── sentiment helpers ──────────────────────────────────────

    def _score(self, text):
        return self.analyzer.polarity_scores(text)['compound']

    def _best_match(self, target, options):
        # Filter out null-like options when we have a meaningful answer
        _null = {'none', 'n/a', 'not applicable', 'nil', 'na', '-'}
        valid = [o for o in options if o.strip().lower() not in _null]
        pool = valid if valid else options

        # If target is numeric, prefer the option containing that number
        t = target.strip()
        if t.isdigit():
            for opt in pool:
                if t in opt.split() or opt.strip() == t:
                    return opt
            # Fallback: pick option whose embedded number is closest
            def _num(s):
                m = re.search(r'\d+', s)
                return int(m.group()) if m else 999
            return min(pool, key=lambda o: abs(_num(o) - int(t)))

        ts = self._score(target)
        return min(pool, key=lambda o: abs(ts - self._score(o)))

    # ── human-like typing ─────────────────────────────────────

    def _type(self, locator, text):
        for ch in text:
            locator.type(ch, delay=random.randint(70, 180))

    # ── chatbot form handler ──────────────────────────────────

    def classify_new_question(self):
        page = self.page
        try:
            cbcn = page.wait_for_selector(".chatbot_MessageContainer", timeout=4000)
            human_delay(1500, 2500)

            while cbcn:
                q_el = page.locator(".botMsg").last
                page.wait_for_timeout(800)
                question = q_el.inner_text().strip()
                self.logger.info(f"  Q: {question}")
                answer = self.model.chatbot_response(question)
                self.logger.info(f"  A: {answer}")

                checkboxes   = cbcn.query_selector_all('input[type="checkbox"]')
                radios       = cbcn.query_selector_all('input[type="radio"]')
                text_input   = page.locator('.chatbot_MessageContainer .textArea')
                chip         = page.query_selector('.chatbot_MessageContainer .chipsContainer .chatbot_Chip')
                suggs        = cbcn.query_selector_all('.ssc__heading')
                dob          = cbcn.query_selector(".dob__container")
                dropdown     = cbcn.query_selector("select")

                if chip:
                    chip.click()
                    human_delay(400, 800)
                    continue

                if radios or checkboxes:
                    buttons = radios or checkboxes
                    ids = [el.evaluate('el => el.id') for el in buttons]
                    best = self._best_match(answer, ids)
                    self.logger.info(f"  Radio/checkbox → {best}")
                    page.locator(f'label[for="{best}"]').click(force=True)

                elif text_input.is_visible():
                    text_input.click()
                    text_input.fill('')
                    self._type(text_input, answer)

                elif dropdown:
                    option_labels = page.eval_on_selector_all(
                        '.chatbot_MessageContainer select option',
                        'els => els.map(e => e.innerText)'
                    )
                    best = self._best_match(answer, option_labels)
                    page.select_option('.chatbot_MessageContainer select', label=best)
                    self.logger.info(f"  Dropdown → {best}")

                elif suggs:
                    options = [el.evaluate('el => el.innerText') for el in suggs]
                    best = self._best_match(answer, options)
                    self.logger.info(f"  Suggestion → {best}")
                    page.click(f'text="{best}"')

                elif dob:
                    parts = answer.strip().split("/")
                    if len(parts) == 3:
                        page.locator("input[name='day']").fill(parts[0])
                        page.locator("input[name='month']").fill(parts[1])
                        page.locator("input[name='year']").fill(parts[2])

                else:
                    self.logger.warning("  No recognised input – exiting chatbot loop")
                    return

                send = page.locator('.sendMsg')
                try:
                    expect(send).to_be_enabled(timeout=3000)
                    human_delay(350, 700)
                    send.click(timeout=3000)
                except Exception:
                    return
                human_delay(800, 1500)

        except Exception as e:
            self.logger.error(f"classify_new_question: {e}")
            return {"response": "error", "error": str(e)}

    # ── standard (non-chatbot) form filler ───────────────────

    def fill_standard_form(self):
        page = self.page
        ui = self.user_info
        self.logger.info("Filling standard form fields...")

        field_map = [
            (["full name", "name"],               ui.get("Full name", "")),
            (["first name", "given name"],        ui.get("First name", "")),
            (["last name", "surname"],             ui.get("Last name", "")),
            (["email"],                            ui.get("Email", "")),
            (["phone", "mobile", "contact"],       ui.get("Mobile", "")),
            (["total experience", "experience"],   ui.get("Total experience", "")),
            (["current salary", "ctc"],            ui.get("Salary", "")),
            (["expected salary", "desired"],       ui.get("Expected salary", "")),
            (["notice period"],                    ui.get("Notice period", "")),
            (["location", "city", "current city"], ui.get("Location", "")),
        ]

        for labels, value in field_map:
            if not value:
                continue
            for lbl in labels:
                filled = False
                for strategy in [
                    f'input[aria-label*="{lbl}" i]',
                    f'input[placeholder*="{lbl}" i]',
                    f'input[name*="{lbl}" i]',
                ]:
                    try:
                        el = page.locator(strategy).first
                        if el.count() > 0 and el.is_visible():
                            el.fill(value)
                            self.logger.info(f"  Filled '{lbl}' → {value}")
                            filled = True
                            break
                    except Exception:
                        continue
                if filled:
                    break

        # Resume upload
        resume = ui.get("Resume path", "")
        if resume and os.path.exists(resume):
            try:
                fi = page.locator('input[type="file"]').first
                if fi.count() > 0:
                    fi.set_input_files(resume)
                    self.logger.info(f"  Resume uploaded: {resume}")
            except Exception as e:
                self.logger.warning(f"  Resume upload failed: {e}")

        # Cover letter
        cover = ui.get("Cover letter", "")
        if cover:
            try:
                ta = page.locator('textarea').first
                if ta.count() > 0 and ta.is_visible():
                    ta.fill(cover)
                    self.logger.info("  Cover letter filled")
            except Exception:
                pass

        human_delay(400, 800)


# ─────────────────────────── main bot ───────────────────────────

class NaukriBot:
    _APPLY_SELECTORS = [
        '#apply-button',
        'button:has-text("Apply")',
        'a:has-text("Apply Now")',
        '.apply-button',
        'button[id*="apply" i]',
        '[data-ga-track*="apply" i]',
    ]
    _SUCCESS_PHRASES = [
        "successfully applied", "application submitted",
        "application sent", "thank you for applying", "applied successfully",
    ]
    _ALREADY_APPLIED = ["already applied", "application submitted", "applied on"]

    def __init__(self, email, password, username, number=10, user_info=None, resume_path="", log_file=None):
        self.usr = [email, password]
        self.username = username
        self.applno = number
        self.applied_count = 0
        self.skipped_count = 0
        self.page_no = 1
        self.base_page_url = ""
        self.tabs = ["profile", "apply", "preference", "similar_jobs"]
        self.tabIndex = 0
        self.tab = "profile"
        self.pattern = re.compile(r'https://.*/myapply/saveApply\?strJobsarr=')
        self.browser = None
        self.page = None
        self.cba = None

        self.user_info = user_info or {}
        if resume_path:
            self.user_info["Resume path"] = resume_path

        self.pw = None  # playwright instance reused across sessions to avoid asyncio-loop conflicts

        log_path = log_file or f"./jab/data/{username}/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger('NaukriBot', log_path)
        self.logger.info(f"NaukriBot ready — user: {username}, target: {number} jobs")

    # ── browser ────────────────────────────────────────────────

    def init_browser(self):
        if self.pw is None:
            self.pw = sync_playwright().start()
        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]
        self.browser = self.pw.chromium.launch(headless=False, args=args)
        ctx = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
        )
        self.page = ctx.new_page()
        self.cba = ChatbotAgent(self.page, self.username, self.user_info, self.logger)
        self.logger.info("Browser launched")

    # ── login ──────────────────────────────────────────────────

    def login(self, retries=3):
        for attempt in range(1, retries + 1):
            try:
                self.logger.info(f"Login attempt {attempt}/{retries}")
                self.page.goto("https://www.naukri.com", timeout=40000)
                human_delay(1000, 2000)
                self.page.click('//*[@id="login_Layer"]')
                human_delay(600, 1000)
                self.page.fill('input[type="text"]', self.usr[0])
                human_delay(300, 600)
                self.page.fill('input[type="password"]', self.usr[1])
                human_delay(300, 600)
                self.page.click('button[type="submit"]')
                self.page.wait_for_url(
                    "https://www.naukri.com/mnjuser/homepage",
                    timeout=20000, wait_until="networkidle"
                )
                self.logger.info("Logged in successfully")
                return True
            except Exception as e:
                self.logger.warning(f"Login attempt {attempt} failed: {e}")
                human_delay(2000, 4000)
        self.logger.error("All login attempts failed")
        return False

    # ── helpers ────────────────────────────────────────────────

    def _click_apply_btn(self):
        for sel in self._APPLY_SELECTORS:
            try:
                btn = self.page.locator(sel).first
                if btn.count() > 0 and btn.is_visible():
                    btn.click()
                    self.logger.info(f"Apply button clicked via: {sel}")
                    return True
            except Exception:
                continue
        return False

    def _already_applied(self):
        try:
            txt = self.page.inner_text('body').lower()
            return any(p in txt for p in self._ALREADY_APPLIED)
        except Exception:
            return False

    def _is_external(self):
        try:
            return 'naukri.com' not in self.page.url
        except Exception:
            return False

    def _save_screenshot(self, label="error"):
        try:
            folder = f"./jab/data/{self.username}/screenshots"
            os.makedirs(folder, exist_ok=True)
            path = f"{folder}/{label}_{datetime.now().strftime('%H%M%S')}.png"
            self.page.screenshot(path=path)
            self.logger.debug(f"Screenshot: {path}")
        except Exception:
            pass

    # ── single-job application ─────────────────────────────────

    def apply_to_job(self, job_url):
        try:
            self.logger.info(f"→ {job_url}")
            self.page.goto(job_url, timeout=30000)
            self.page.wait_for_load_state('networkidle', timeout=15000)
            human_delay(700, 1400)

            if self._already_applied():
                self.logger.info("  Already applied — skipping")
                self.skipped_count += 1
                return False

            if not self._click_apply_btn():
                self.logger.warning("  No apply button found — skipping")
                return False

            human_delay(1500, 2500)

            if self._is_external():
                self.logger.info("  External site — skipping")
                self.skipped_count += 1
                self.page.go_back()
                human_delay(1000, 1500)
                return False

            # Handle chatbot form
            try:
                self.page.wait_for_selector(".chatbot_MessageContainer", timeout=3000)
                self.logger.info("  Chatbot form detected")
                self.cba.classify_new_question()
                human_delay(800, 1500)
            except PlaywrightTimeoutError:
                pass

            # Handle standard application form pages (multi-step)
            max_steps = 5
            for step in range(max_steps):
                try:
                    form = self.page.locator('form').first
                    if form.count() > 0 and form.is_visible():
                        self.logger.info(f"  Standard form step {step + 1}")
                        self.cba.fill_standard_form()
                        # Also handle any chatbot widget inside this page
                        try:
                            self.page.wait_for_selector(".chatbot_MessageContainer", timeout=1500)
                            self.cba.classify_new_question()
                        except PlaywrightTimeoutError:
                            pass
                        # Click next/submit button
                        advanced = False
                        for btn_text in ["Submit", "Apply", "Save & Apply", "Continue", "Next"]:
                            try:
                                btn = self.page.locator(f'button:has-text("{btn_text}")').last
                                if btn.count() > 0 and btn.is_visible():
                                    btn.click()
                                    self.logger.info(f"  Clicked '{btn_text}'")
                                    human_delay(1200, 2000)
                                    advanced = True
                                    break
                            except Exception:
                                continue
                        if not advanced:
                            break
                    else:
                        break
                except Exception:
                    break

            # Confirm success
            try:
                expect(self.page).to_have_url(self.pattern, timeout=4000)
                self.logger.info("  ✓ Applied (URL pattern)")
                return True
            except Exception:
                pass

            try:
                body = self.page.inner_text('body').lower()
                if any(p in body for p in self._SUCCESS_PHRASES):
                    self.logger.info("  ✓ Applied (success text)")
                    return True
            except Exception:
                pass

            self.logger.info("  ✓ Applied (assumed)")
            return True

        except Exception as e:
            self.logger.error(f"  apply_to_job error: {e}")
            self._save_screenshot("apply_error")
            return False

    # ── job link extraction ────────────────────────────────────

    def get_job_links(self):
        try:
            self.page.wait_for_load_state('networkidle', timeout=15000)
            # Scroll to load lazy content
            for pct in [0.3, 0.6, 1.0]:
                self.page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
                human_delay(400, 700)

            links = self.page.eval_on_selector_all(
                '.title, .jobTitle, [class*="job-title"] a',
                'els => els.map(e => e.getAttribute("href")).filter(h => h && h.includes("naukri.com"))'
            )
            if not links:
                links = self.page.eval_on_selector_all(
                    'a[href*="/job-listings-"]',
                    'els => els.map(e => e.href)'
                )
            self.logger.info(f"Found {len(links)} jobs on page {self.page_no}")
            return links
        except Exception as e:
            self.logger.error(f"get_job_links: {e}")
            return []

    # ── paginated apply loop ───────────────────────────────────

    def apply_(self):
        links = self.get_job_links()

        for jl in links:
            if self.applied_count >= self.applno:
                self.logger.info(f"Target reached ({self.applied_count}/{self.applno})")
                return
            human_delay(1500, 3000)
            if self.apply_to_job(jl):
                self.applied_count += 1
                self.logger.info(f"Progress: {self.applied_count}/{self.applno}")
            # Return to search results
            try:
                self.page.go_back(timeout=10000)
                self.page.wait_for_load_state('domcontentloaded', timeout=10000)
                human_delay(600, 1000)
            except Exception:
                try:
                    self.page.goto(self.base_page_url)
                    self.page.wait_for_load_state('domcontentloaded')
                except Exception:
                    pass

        # Paginate
        if self.applied_count < self.applno:
            self.page_no += 1
            parsed = urlparse(self.base_page_url)
            clean_path = re.sub(r'-\d+$', '', parsed.path)
            new_url = urlunparse(parsed._replace(path=f"{clean_path}-{self.page_no}"))
            self.logger.info(f"Paginating → page {self.page_no}")
            try:
                self.page.goto(new_url, timeout=20000)
                self.page.wait_for_load_state('networkidle', timeout=15000)
                self.apply_()
            except Exception as e:
                self.logger.warning(f"Pagination stopped: {e}")

    # ── multi-role single-session apply ───────────────────────

    def multi_role_apply(self, roles, experience='1to5', location='', job_age='3'):
        """
        Search for all roles in one browser session (no re-login).
        Caller is responsible for init_browser() + login() before calling this,
        and shutdown() after.
        """
        self.search = roles
        self.experience = experience
        self.location = location
        self.jobage = job_age
        human_delay(800, 1500)
        self.filter_()
        self.base_page_url = self.page.url
        self.apply_()
        result = {
            "response": "done",
            "applied": self.applied_count,
            "skipped": self.skipped_count,
        }
        self.logger.info(f"Session complete: {result}")
        return result

    # ── filter mode (manual --filters, preserves legacy behaviour) ─

    def filter_apply(self, s, e='', l='', ja='3'):
        if not s:
            self.logger.error("Search keyword is required")
            return {"response": "error: no search keyword", "applied": 0}
        self.search = s
        self.experience = e
        self.location = l
        self.jobage = ja
        self.init_browser()
        if not self.login():
            return {"response": "login failed", "applied": 0}
        human_delay(800, 1500)
        self.filter_()
        self.base_page_url = self.page.url
        self.apply_()
        result = {
            "response": "done",
            "applied": self.applied_count,
            "skipped": self.skipped_count,
        }
        self.logger.info(f"Session complete: {result}")
        self.close()
        return result

    def filter_(self):
        try:
            srch = self.page.locator(".nI-gNb-sb__icon-wrapper")
            srch.click()
            human_delay(500, 900)

            kw_input = self.page.locator('input[placeholder="Enter keyword / designation / companies"]')
            kw_input.click()
            kw_input.fill('')

            # Support list of keywords — each pressed as a chip with Enter
            keywords = self.search if isinstance(self.search, list) else [self.search]
            for kw in keywords:
                kw_input.fill(kw)
                human_delay(300, 500)
                kw_input.press('Enter')
                human_delay(300, 500)

            if self.location:
                loc_input = self.page.locator('input[placeholder="Enter location"]')
                loc_input.click()
                loc_input.fill(self.location)
                human_delay(200, 400)
                loc_input.press('Enter')
                human_delay(200, 400)

            if self.experience:
                # Accept "1to5" range → use the min value index; also accept plain int
                exp_str = str(self.experience)
                min_exp = int(exp_str.split('to')[0]) if 'to' in exp_str else int(exp_str)
                try:
                    self.page.locator('#experienceDD').click()
                    human_delay(300, 600)
                    self.page.locator(f'li[index="{min_exp}"]').click()
                    human_delay(200, 400)
                except Exception:
                    pass

            srch.click()
            self.page.wait_for_load_state('load')

            if self.jobage:
                sep = '&' if '?' in self.page.url else '?'
                self.page.goto(self.page.url + f"{sep}jobAge={self.jobage}")
                self.page.wait_for_load_state('networkidle')
        except Exception as e:
            self.logger.error(f"filter_ error: {e}")

    # ── checkbox / tab mode ────────────────────────────────────

    def checkbox_apply(self):
        try:
            cbs = self.page.locator('.naukicon-ot-checkbox').element_handles()
            self.logger.info(f"Checkboxes found: {len(cbs)}")
            if not cbs:
                return {"status": "finished", "found": 0, "clicked": 0}

            clicked = 0
            for cb in cbs[:5]:
                try:
                    cb.click()
                    clicked += 1
                    human_delay(200, 500)
                except Exception:
                    continue

            self.page.locator('.multi-apply-button').click()
            human_delay(1500, 2500)

            try:
                expect(self.page.locator(".chatbot_MessageContainer")).to_be_visible(timeout=3000)
                return {"status": "underway", "found": len(cbs), "clicked": clicked}
            except Exception:
                try:
                    expect(self.page).to_have_url(self.pattern, timeout=3000)
                    return {"status": "done", "clicked": clicked}
                except Exception:
                    raise
        except Exception:
            return {"status": "failed"}

    def start_apply(self, tab):
        self.tabIndex = 0
        self.tab = tab
        self.init_browser()
        if self.login():
            return self.bot_actions()

    def bot_actions(self):
        try:
            human_delay(2000, 3000)
            self.page.click('.nI-gNb-menuItems__anchorDropdown')
            if self.tab != "profile":
                self.page.click(f"#{self.tab}")
            self.page.wait_for_load_state("networkidle")

            if self.applied_count >= self.applno:
                self.logger.info(f"Target reached: {self.applied_count} jobs")
                return {"response": "done", "applied": self.applied_count}

            cbapl = self.checkbox_apply()

            if cbapl["status"] == 'failed':
                self.logger.info("Daily quota exhausted")
                self.close()
                return {"response": "quota finished", "applied": self.applied_count}

            if cbapl["status"] == 'done':
                self.applied_count += cbapl["clicked"]
                return self.bot_actions()

            if cbapl["status"] == 'underway':
                self.cba.classify_new_question()
                try:
                    expect(self.page).to_have_url(self.pattern, timeout=5000)
                    self.applied_count += cbapl["clicked"]
                    return self.bot_actions()
                except Exception as e:
                    self.logger.error(f"bot_actions underway error: {e}")
                    self.close()
                    return {"response": "error", "error": str(e)}

            if cbapl['status'] == "finished":
                if self.tabIndex + 1 < len(self.tabs):
                    self.tabIndex += 1
                    self.tab = self.tabs[self.tabIndex]
                    return self.bot_actions()
                else:
                    self.logger.info("All tabs exhausted")
                    self.close()
                    return {"response": "all tabs done", "applied": self.applied_count}

        except Exception as e:
            self.close()
            self.logger.error(f"bot_actions error (applied {self.applied_count}): {e}")
            return {"response": "error", "error": str(e)}

    def close(self):
        if self.browser:
            self.browser.close()
            self.browser = None
            self.logger.info("Browser closed")

    def shutdown(self):
        """Call once after all roles are done to fully stop playwright."""
        self.close()
        if self.pw:
            self.pw.stop()
            self.pw = None
