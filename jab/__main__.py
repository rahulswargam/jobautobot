import os
import argparse
import json
from getpass import getpass
from .modules.naukri import NaukriBot
from .modules.model import ChatbotBuild

BANNER = """
╔══════════════════════════════════════════╗
║           Naukri Job Auto-Apply          ║
║         github.com/aman-dayal/jobautobot ║
╚══════════════════════════════════════════╝
"""


def load_user_info(username):
    """Load user profile from disk if it exists."""
    path = f"./jab/data/{username}/profile.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Fall back to base template
    with open("./jab/data/user_data.json") as f:
        return json.load(f)


def save_user_info(username, data):
    os.makedirs(f"./jab/data/{username}", exist_ok=True)
    with open(f"./jab/data/{username}/profile.json", "w") as f:
        json.dump(data, f, indent=4)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="Naukri Job Auto-Apply Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m jab --email you@email.com --train
  python -m jab --email you@email.com --apply --filters
  python -m jab --email you@email.com --apply
  python -m jab --email you@email.com --apply --filters --resume /path/to/resume.pdf
        """,
    )
    parser.add_argument("--email",   required=True, help="Naukri account email")
    parser.add_argument("--train",   action="store_true", help="Train the ML model")
    parser.add_argument("--apply",   action="store_true", help="Start applying to jobs")
    parser.add_argument("--filters", action="store_true", help="Use search filters")
    parser.add_argument("--resume",  default="",          help="Path to resume file (PDF/DOC)")
    parser.add_argument("--jobs",    type=int, default=0,  help="Number of jobs to apply to")
    args = parser.parse_args()

    email    = args.email
    username = email

    # ── TRAIN MODE ────────────────────────────────────────────
    if args.train:
        print("[Train] Building and training the ML model...")
        chb = ChatbotBuild(username)
        chb.train_model()
        # Save training data for runtime use
        out_dir = f"./jab/data/{username}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/training_data.json", "w") as f:
            json.dump(chb.training_data, f, indent=4)
        print(f"[Train] Done. Model saved to ./jab/data/{username}/model.keras")
        return

    # ── APPLY MODE ────────────────────────────────────────────
    if args.apply:
        password = getpass("Naukri Password: ")
        number   = args.jobs or int(input("Number of jobs to apply (default 10): ").strip() or 10)
        user_info = load_user_info(username)

        resume = args.resume
        if not resume:
            resume = input("Path to resume file (leave blank to skip upload): ").strip()

        nb = NaukriBot(
            email=email,
            password=password,
            username=username,
            number=number,
            user_info=user_info,
            resume_path=resume,
        )

        if args.filters:
            print("\nEnter search filters (press Enter to skip optional fields):")
            search     = input("Job keyword / designation (required): ").strip()
            if not search:
                print("Error: search keyword is required with --filters")
                return
            experience = input("Years of experience filter (e.g. 2, optional): ").strip()
            location   = input("Preferred location (optional): ").strip()
            job_age    = input("Max job posting age in days (default 3): ").strip()

            experience = int(experience) if experience.isdigit() else ''
            job_age    = int(job_age)    if job_age.isdigit()    else '3'

            result = nb.filter_apply(search, experience, location, job_age)
            print(f"\nResult: {result}")
        else:
            print(f"\nAvailable tabs: {nb.tabs}")
            tab = input("Choose a tab to start from (default: profile): ").strip() or "profile"
            if tab not in nb.tabs:
                print(f"Invalid tab. Choose from: {nb.tabs}")
                return
            result = nb.start_apply(tab)
            print(f"\nResult: {result}")
        return

    # ── NO MODE SELECTED ──────────────────────────────────────
    parser.print_help()
    print("\nUse --train to train the model or --apply to start applying.")


if __name__ == "__main__":
    main()
