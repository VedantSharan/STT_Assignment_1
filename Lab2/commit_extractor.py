from pydriller import Repository
import re
import csv
from datetime import datetime

class BugFixCommitIdentifier:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.bug_keywords = [
            'fix', 'bug', 'issue', 'error', 'problem', 'resolve', 'correct',
            'repair', 'patch', 'hotfix', 'bugfix', 'defect', 'fault'
        ]
        self.bug_patterns = [
            r'\bfix(?:es|ed|ing)?\b',
            r'\bbug(?:s)?\b',
            r'\bissue(?:s)?\b',
            r'\berror(?:s)?\b',
            r'\bresolv(?:e|es|ed|ing)\b',
            r'\bcorrect(?:s|ed|ing)?\b',
            r'\brepair(?:s|ed|ing)?\b',
            r'\bpatch(?:es|ed|ing)?\b'
        ]
    
    def is_bug_fix_commit(self, commit):
        """
        Determine if a commit is a bug-fixing commit based on multiple criteria
        """
        message = commit.msg.lower()
        keyword_match = any(keyword in message for keyword in self.bug_keywords)
        pattern_match = any(re.search(pattern, message, re.IGNORECASE) 
                          for pattern in self.bug_patterns)

        exclude_patterns = [
            r'\badd(?:s|ed|ing)?\b.*feature',
            r'\bimplement(?:s|ed|ing)?\b',
            r'\brefactor(?:s|ed|ing)?\b',
            r'\bupdate(?:s|ed|ing)?\b.*version',
            r'\bmerge\b',
            r'\binitial commit\b'
        ]
        
        exclude_match = any(re.search(pattern, message, re.IGNORECASE) 
                           for pattern in exclude_patterns)
        
        return (keyword_match or pattern_match) and not exclude_match
    
    def safe_get_parent_hashes(self, commit):
        try:
            if hasattr(commit, 'parents'):
                parents = commit.parents
                if parents:
                    # Check if parents are objects or strings
                    if isinstance(parents[0], str):
                        return parents
                    else:
                        return [parent.hash for parent in parents]
                else:
                    return []
            else:
                return []
        except Exception as e:
            print(f"Warning: Could not get parent hashes for commit {commit.hash}: {e}")
            return []
    
    def safe_get_modified_files(self, commit):
        try:
            if hasattr(commit, 'modified_files') and commit.modified_files:
                return [mod.filename for mod in commit.modified_files if mod.filename]
            else:
                return []
        except Exception as e:
            print(f"Warning: Could not get modified files for commit {commit.hash}: {e}")
            return []
    
    def extract_bug_fix_commits(self, max_commits=2000):
        bug_fix_commits = []
        commit_count = 0
        
        print("Analyzing commits...")      
        try:
            for commit in Repository(self.repo_path).traverse_commits():
                commit_count += 1
                if commit_count > max_commits:
                    break              
                if self.is_bug_fix_commit(commit):
                    parent_hashes = self.safe_get_parent_hashes(commit)
                    modified_files = self.safe_get_modified_files(commit)               
                    commit_info = {
                        'hash': commit.hash,
                        'message': commit.msg.replace('\n', ' ').replace('\r', ' ').strip(),
                        'parent_hashes': parent_hashes,
                        'is_merge': len(parent_hashes) > 1,
                        'modified_files': modified_files,
                        
                    }
                    bug_fix_commits.append(commit_info)
                
                if commit_count % 100 == 0:
                    print(f"Processed {commit_count} commits, found {len(bug_fix_commits)} bug fixes")
                    
        except Exception as e:
            print(f"Error during commit traversal: {e}")
            print("This might be due to repository issues or PyDriller version compatibility")
        
        print(f"Total commits analyzed: {commit_count}")
        print(f"Bug-fixing commits found: {len(bug_fix_commits)}")
        
        return bug_fix_commits
    
    def save_bug_fix_commits(self, commits, filename='bug_fix_commits_01.csv'):
        """
        Save bug-fixing commits to CSV
        """
        if not commits:
            print("No commits to save!")
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['hash', 'message', 'parent_hashes', 'is_merge', 
                             'modified_files']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for commit in commits:
                    # Convert lists to string representation for CSV
                    commit_copy = commit.copy()
                    commit_copy['parent_hashes'] = ';'.join(commit['parent_hashes']) if commit['parent_hashes'] else ''
                    commit_copy['modified_files'] = ';'.join(commit['modified_files']) if commit['modified_files'] else ''
                    writer.writerow(commit_copy)
            
            print(f"Bug-fixing commits saved to {filename}")
            
        except Exception as e:
            print(f"Error saving commits to CSV: {e}")


if __name__ == "__main__":

    REPO_PATH = "graphics"
    

    identifier = BugFixCommitIdentifier(REPO_PATH)
    bug_commits = identifier.extract_bug_fix_commits()
    identifier.save_bug_fix_commits(bug_commits)