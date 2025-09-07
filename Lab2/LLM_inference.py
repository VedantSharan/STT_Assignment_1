import csv
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import re

class CommitLLMInference:
    def __init__(self, model_name="mamiksik/CommitPredictorT5"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the CommitPredictorT5 model"""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback model...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
                self.model.to(self.device)
                print("Fallback model loaded")
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                self.model = None
                self.tokenizer = None
    
    def prepare_input_text(self, diff_content, filename):
        if not diff_content:
            return f"fix {filename}"
        
        # Clean and truncate diff
        diff_lines = diff_content.split('\n')
        relevant_lines = []
        
        for line in diff_lines:
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                relevant_lines.append(line)
                if len(relevant_lines) >= 20:  # Limit to 20 lines
                    break
        
        if not relevant_lines:
            return f"fix {filename}"
        
        diff_text = '\n'.join(relevant_lines)
        if len(diff_text) > 400:
            diff_text = diff_text[:400]
        
        return diff_text
    
    def generate_commit_message(self, diff_content, filename):

        # if not self.model or not self.tokenizer:
        #     return self.fallback_classification(diff_content, filename)
        
        # try:
        input_text = self.prepare_input_text(diff_content, filename)
        
        # Tokenize
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=30,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generated_text = generated_text.strip()
        if len(generated_text) > 100:
            generated_text = generated_text[:100]
        
        return generated_text if generated_text else self.fallback_classification(diff_content, filename)
            
        # except Exception as e:
        #     print(f"Error in generation: {e}")
        #     return self.fallback_classification(diff_content, filename)
    
    # def fallback_classification(self, diff_content, filename):
    #     """Simple rule-based fallback when model fails"""
    #     if not diff_content:
    #         return f"update {filename}"
        
    #     diff_lower = diff_content.lower()
        
    #     # Simple keyword matching
    #     if any(word in diff_lower for word in ['fix', 'bug', 'error']):
    #         return "fix bug"
    #     elif any(word in diff_lower for word in ['add', 'new', 'create']):
    #         return "add feature"
    #     elif any(word in diff_lower for word in ['remove', 'delete', 'clean']):
    #         return "remove code"
    #     elif any(word in diff_lower for word in ['update', 'change', 'modify']):
    #         return "update code"
    #     else:
    #         return "fix issue"
    
    def classify_fix_type(self, generated_message, diff_content, filename):
        """Extract fix type from generated message"""
        if not generated_message:
            return "general"
        
        text = f"{generated_message} {diff_content} {filename}".lower()
        
        # Define fix type patterns
        patterns = {
            'syntax': ['syntax', 'parse', 'import', 'compilation'],
            'logic': ['logic', 'condition', 'algorithm', 'if', 'else', 'loop'],
            'performance': ['performance', 'optimize', 'speed', 'efficient'],
            'security': ['security', 'auth', 'permission', 'validate'],
            'ui': ['ui', 'display', 'render', 'style', 'css', 'html'],
            'api': ['api', 'endpoint', 'request', 'response', 'http'],
            'database': ['database', 'query', 'sql', 'table', 'schema'],
            'null': ['null', 'none', 'empty', 'undefined', 'missing'],
            'exception': ['exception', 'error', 'try', 'catch', 'throw'],
            'test': ['test', 'spec', 'unit', 'assert', 'mock'],
            'refactor': ['refactor', 'clean', 'structure', 'organize'],
            'dependency': ['dependency', 'package', 'library', 'version']
        }
        
        # Count matches
        scores = {}
        for fix_type, keywords in patterns.items():
            score = sum(text.count(keyword) for keyword in keywords)
            scores[fix_type] = score
        
        # Return highest scoring type or 'general'
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return "general"
    
    def process_csv_file(self, input_filename, output_filename=None):
        """Process CSV file and add LLM inference"""
        if not output_filename:
            output_filename = input_filename.replace('.csv', '_with_llm.csv')
        
        records = []
        
        # Read CSV
        with open(input_filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            
            print(f"Processing {input_filename} with T5 model...")
            
            for i, row in enumerate(reader):
                diff_content = row.get('Diff', '')
                filename = row.get('Filename', '')
                
                # Generate commit message using T5
                generated_message = self.generate_commit_message(diff_content, filename)
                
                # Classify fix type based on generated message
                fix_type = self.classify_fix_type(generated_message, diff_content, filename)
                
                # Update row
                row['LLM Inference (fix type)'] = f"{generated_message}({fix_type})"
                records.append(row)
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1} records...")
                
                # Print sample for first few
                if i < 3:
                    print(f"Sample {i+1}: {filename} -> {fix_type}")
        

        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"Completed. Results saved to {output_filename}")
        

        from collections import Counter
        fix_types_count = Counter(
            match.group(1)
            for record in records
            # This finds the pattern and assigns the match object, skipping if no match is found
            if (match := re.search(r'\((\w+)\)', record.get('LLM Inference (fix type)', '')))
        )
        print(f"\nFix type distribution:")
        for fix_type, count in fix_types_count.most_common():
            print(f"  {fix_type}: {count}")
        
        return records

def main():
    

    input_file = "diff_analysis_01.csv"
    

    output_file = "diff_analysis_01_with_llm.csv"

    try:
        inference = CommitLLMInference()
        if inference.model is None:
            print("Warning: Model not loaded, results may be less accurate")
        
        records = inference.process_csv_file(input_file, output_file)
        print(f"\nProcessed {len(records)} records successfully.")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()