from pydriller import Repository
import csv
import os
import pandas as pd

class DiffExtractor:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        
    def clean_text(self, text):
        if not text:
            return ""
        # Normalize newlines and remove them for a single-line message
        cleaned_text = str(text).replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        return cleaned_text

    def extract_diffs(self, bug_commits_hashes):
        all_diff_data = []
        print(f"Processing {len(bug_commits_hashes)} bug-fixing commits...")

        for i, commit_hash in enumerate(bug_commits_hashes):
            print(f"  [{i+1}/{len(bug_commits_hashes)}] Processing commit: {commit_hash}")
            

            repo_commit = list(Repository(self.repo_path, single=commit_hash).traverse_commits())[0]

            if not repo_commit.modified_files:
                continue

            commit_message = self.clean_text(repo_commit.msg)

            

            for mod in repo_commit.modified_files:
                try:
                    source_before = mod.source_code_before
                except Exception:
                    source_before = ""
                try:
                    source_code = mod.source_code
                except Exception:
                    source_code = ""
                record = {
                    'Hash': repo_commit.hash,
                    'Message': commit_message,
                    'Filename': mod.filename,
                    'Source Code (before)': self.clean_text(source_before),
                    'Source Code (current)': self.clean_text(source_code),
                    'Diff': self.clean_text(mod.diff),
                    'LLM Inference (fix type)': '', # Placeholder
                    'Rectified Message': ''       # Placeholder
                }
                all_diff_data.append(record)


        print(f" Finished processing. Found {len(all_diff_data)} file changes.")
        return all_diff_data

    def save_to_csv(self, diff_data, filename='diff_analysis_6.csv'):
        if not diff_data:
            print("No data to save.")
            return

        fieldnames = [
            'Hash', 'Message', 'Filename', 'Source Code (before)', 
            'Source Code (current)', 'Diff', 'LLM Inference (fix type)', 
            'Rectified Message'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(diff_data)
        
        print(f" Successfully saved {len(diff_data)} records to '{filename}'")

def load_bug_commits_from_csv(filename='bug_fix_commits.csv'):

    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        hashes = [row['hash'] for row in reader]
        print(f" Loaded {len(hashes)} commit hashes from '{filename}'")
        return hashes

def main():

    REPO_PATH = "graphics"
    INPUT_CSV = "bug_fix_commits_01.csv"
    OUTPUT_CSV = "diff_analysis_01.csv"
    # COMMIT_LIMIT = 20 

    bug_commits = load_bug_commits_from_csv(INPUT_CSV)
    if not bug_commits:
        return

    extractor = DiffExtractor(REPO_PATH)
    diff_data = extractor.extract_diffs(bug_commits)

    if diff_data:
        extractor.save_to_csv(diff_data, OUTPUT_CSV)
    
    print("\n Process completed.")

if __name__ == "__main__":
    main()