# Naukri Job Auto-Apply Bot

Automatically applies to DevOps, SRE, and Cloud jobs on Naukri.com.
It opens a browser, logs into your account, fills forms, answers chatbot questions, and submits applications — all on its own.

---

## Before You Start

Make sure you have:
- Mac with Python 3.9
- A Naukri.com account
- Your resume saved as a PDF

---

## One-Time Setup (Do This Once)

### 1. Go to the project folder

```bash
cd ~/Desktop/jobautobot
```

### 2. Install required packages

```bash
pip3 install -r requirements.txt
```

### 3. Install the browser the bot uses

```bash
python3 -m playwright install chromium
```

### 4. Add your mobile number and date of birth

Open this file:

```
jab/data/user_data.json
```

Find these two lines and fill them in:

```json
"Mobile": "your 10-digit number",
"Date of birth": "DD/MM/YYYY"
```

Everything else (name, email, skills, salary, location) is already filled in.

---

## Every Time You Want to Apply

### Step 1 — Train the bot (run once, or after updating your details)

```bash
python3 -m jab --email swargamrahul02@gmail.com --train
```

This teaches the bot how to answer application questions using your profile.
It saves a model file to `jab/data/swargamrahul02@gmail.com/model.keras`.

---

### Step 2 — Start applying

```bash
python3 -m jab --email swargamrahul02@gmail.com --apply
```

It will ask:

```
Naukri Password:                        ← type your Naukri password (hidden)
Number of jobs to apply (default 10):   ← type a number or press Enter for 10
```

A browser window opens and the bot starts applying automatically.

---

## What the Bot Does Automatically

- Searches for DevOps / SRE / Cloud jobs in Hyderabad
- Opens each job listing and clicks Apply
- Fills out every form field using your profile
- Answers all chatbot questions on its own
- Moves to the next job when done

### Roles it searches (auto):
DevOps Engineer, SRE, Site Reliability Engineer, Cloud Engineer, Platform Engineer, AWS DevOps, Kubernetes Engineer, Infrastructure Engineer

### Questions it auto-answers:

| Question | Answer |
|---|---|
| Years of experience in any tool/skill | 4 |
| Have you implemented / used this tool? | Yes |
| Have you worked with this company before? | No |
| Last working day? | 30 days from today |
| Open to relocate? | Yes |
| Immediately available? | Yes |
| Notice period | 30 days |
| Current CTC | 9 LPA |
| Expected CTC | 15 LPA |
| Preferred locations | Hyderabad, Bengaluru, Pune, Chennai, Mumbai |

---

## Other Useful Commands

### Apply to more jobs at once

```bash
python3 -m jab --email swargamrahul02@gmail.com --apply --jobs 50
```

Applies to 50 jobs without asking for the number.

---

### Search for a specific job title

```bash
python3 -m jab --email swargamrahul02@gmail.com --apply --filters
```

Then type in whatever role you want:

```
Job keyword / designation: DevOps Engineer
Years of experience filter: 5
Preferred location: Bangalore
Max job posting age in days: 3
```

---

## Where Are My Logs?

Every session saves a log so you can see what was applied:

```
jab/data/swargamrahul02@gmail.com/session_YYYYMMDD_HHMMSS.log
```

To view the latest log:

```bash
ls -t jab/data/swargamrahul02@gmail.com/*.log | head -1 | xargs cat
```

---

## Something Went Wrong?

| Problem | Fix |
|---|---|
| `No module named 'playwright'` | Run `pip3 install -r requirements.txt` |
| `Executable doesn't exist` | Run `python3 -m playwright install chromium` |
| `model.keras not found` | Run `--train` first |
| Login failed | Double-check your Naukri password |
| 0 jobs applied | Try `--filters` with a specific keyword |

---

## Quick Reference

| Command | What it does |
|---|---|
| `pip3 install -r requirements.txt` | Install packages (once only) |
| `python3 -m playwright install chromium` | Install browser (once only) |
| `python3 -m jab --email ... --train` | Train the bot with your profile |
| `python3 -m jab --email ... --apply` | Auto-apply to DevOps/SRE jobs |
| `python3 -m jab --email ... --apply --jobs 50` | Apply to 50 jobs |
| `python3 -m jab --email ... --apply --filters` | Search a specific job title |
