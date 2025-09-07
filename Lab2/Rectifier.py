import pandas as pd
import re

def rectify_commit_messages(csv_file):
    df = pd.read_csv(csv_file)
    
    def get_fix_type(llm_message):
        match = re.search(r'\((.*?)\)', str(llm_message))
        return match.group(1) if match else 'general'
    
    def rectify_message(original, llm_msg, filename):
        fix_type = get_fix_type(llm_msg)
        clean_llm = re.sub(r'\s*\([^)]*\)', '', str(llm_msg)).strip()
        
        # Simple rectification rules
        if len(original.split()) > 4 and fix_type.lower() in original.lower():
            return original  # Keep if already good
        
        if len(clean_llm) > 10:
            return clean_llm  # Use LLM if descriptive
        
        # Generate based on context
        file_type = filename.split('.')[-1] if '.' in filename else 'file'
        return f"Fix {fix_type} issue in {file_type}"
    
    # Apply rectification
    df['Rectified Message'] = df.apply(
        lambda row: rectify_message(row['Message'], row['LLM Inference (fix type)'], row['Filename']), 
        axis=1
    )
    
    return df


# Run the rectifier
df = rectify_commit_messages('diff_analysis_01_with_llm.csv')
df.to_csv('rectified_output.csv', index=False)
print("\nRectified CSV saved as 'rectified_output.csv'")