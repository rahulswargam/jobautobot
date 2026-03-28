# Naukri Job Auto-Apply Bot

Automatically applies to DevOps, SRE, and Cloud jobs on Naukri.com.
It opens a browser, logs into your account, fills forms, answers chatbot questions, and submits applications — all on its own.

---

## Before You Start

Make sure you have:
- Mac / Linux / Windows with Python 3.9+
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

### 4. Fill in your details

Open this file:

```
jab/data/user_data.json
```

Fill in all the fields with your own details:

```json
{
    "Salutation": "Mr / Ms / Mrs",
    "Full name": "Your Full Name",
    "First name": "Your First Name",
    "Last name": "Your Last Name",
    "Location": "Your Current City",
    "Current location": "Your Current City",
    "Total experience": "5",
    "Current CTC": "your current salary in LPA",
    "Expected salary": "your expected salary in LPA",
    "Salary": "your expected salary in LPA",
    "Mobile": "your 10-digit mobile number",
    "Email": "your-email@gmail.com",
    "Notice period": "30",
    "Date of birth": "DD/MM/YYYY",
    "Gender": "Male / Female",
    "Preferred work location": "City1, City2, City3",
    "Relocate": "Yes",
    "Open to relocate": "Yes",
    "Immediately available": "Yes",
    "Highest qualification": "B.Tech / MCA / B.Sc",
    "Resume path": "/full/path/to/your/resume.pdf",
    "Skills": {
        "AWS": "4",
        "Docker": "3",
        "Kubernetes": "3",
        "Linux": "5"
    }
}
```

> Add or remove skills as needed. The number next to each skill is your years of experience in it.

---

## Every Time You Want to Apply

### Step 1 — Train the bot

Run this once after filling in your details, or again whenever you update them.

```bash
python3 -m jab --email your-email@gmail.com --train
```

Replace `your-email@gmail.com` with your actual Naukri account email.

This teaches the bot how to answer application questions using your profile.
It saves the model to `jab/data/your-email@gmail.com/model.keras`.

---

### Step 2 — Start applying

```bash
python3 -m jab --email your-email@gmail.com --apply
```

It will ask:

```
Naukri Password:                        ← type your Naukri password (hidden)
Number of jobs to apply (default 10):   ← type a number or press Enter for 10
```

A browser window opens and the bot starts applying automatically.

---

## What the Bot Does Automatically

- Searches for DevOps / SRE / Cloud jobs in your preferred location
- Opens each job listing and clicks Apply
- Fills out every form field using your profile
- Answers all chatbot questions on its own
- Moves to the next job when done

### Roles it searches (auto):
DevOps Engineer, SRE, Site Reliability Engineer, Cloud Engineer, Platform Engineer, AWS DevOps, Kubernetes Engineer, Infrastructure Engineer

### Questions it auto-answers:

| Question | Answer |
|---|---|
| Years of experience in any tool/skill | From your Skills in `user_data.json` |
| Have you implemented / used this tool? | Yes |
| Have you worked with this company before? | No |
| Last working day? | 30 days from today (auto-calculated) |
| Open to relocate? | Yes |
| Immediately available? | Yes |
| Notice period | From your `user_data.json` |
| Current CTC | From your `user_data.json` |
| Expected CTC | From your `user_data.json` |
| Preferred locations | From your `user_data.json` |

---

## Other Useful Commands

### Apply to more jobs at once

```bash
python3 -m jab --email your-email@gmail.com --apply --jobs 50
```

Applies to 50 jobs without asking for the number.

---

### Search for a specific job title

```bash
python3 -m jab --email your-email@gmail.com --apply --filters
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
jab/data/your-email@gmail.com/session_YYYYMMDD_HHMMSS.log
```

To view the latest log:

```bash
ls -t jab/data/your-email@gmail.com/*.log | head -1 | xargs cat
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
| `python3 -m jab --email your-email@gmail.com --train` | Train the bot with your profile |
| `python3 -m jab --email your-email@gmail.com --apply` | Auto-apply to DevOps/SRE jobs |
| `python3 -m jab --email your-email@gmail.com --apply --jobs 50` | Apply to 50 jobs |
| `python3 -m jab --email your-email@gmail.com --apply --filters` | Search a specific job title |
