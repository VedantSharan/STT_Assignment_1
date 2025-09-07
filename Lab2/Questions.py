import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

class SimpleCommitAnalyzer:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
    
    def prepare_data(self):
        # Extract fix types
        self.df['Fix_Type'] = self.df['LLM Inference (fix type)'].apply(self.extract_fix_type)
        
        # Calculate message lengths
        self.df['Message_Length'] = self.df['Message'].apply(lambda x: len(str(x).split()))
        
        # Clean LLM messages
        self.df['LLM_Clean'] = self.df['LLM Inference (fix type)'].apply(self.clean_llm_message)
        self.df['LLM_Length'] = self.df['LLM_Clean'].apply(lambda x: len(str(x).split()))
        
        # Extract file extensions
        self.df['File_Extension'] = self.df['Filename'].apply(self.get_file_extension)
        
        # Apply rectification
        self.df['Rectified Message'] = self.df.apply(self.rectify_message, axis=1)
        self.df['Rectified_Length'] = self.df['Rectified Message'].apply(lambda x: len(str(x).split()))
    
    def extract_fix_type(self, llm_message):
        match = re.search(r'\((.*?)\)', str(llm_message))
        return match.group(1) if match else 'unknown'
    
    def clean_llm_message(self, llm_message):
        return re.sub(r'\s*\([^)]*\)', '', str(llm_message)).strip()
    
    def get_file_extension(self, filename):
        return filename.split('.')[-1] if '.' in str(filename) else 'unknown'
    
    def rectify_message(self, row):
        fix_type = row['Fix_Type']
        clean_llm = row['LLM_Clean']
        original = str(row['Message'])
        
        # Keep original if precise
        if self.is_precise_message(original, fix_type):
            return original
        
        # Use LLM if good quality
        if len(clean_llm) > 10 and self.is_meaningful(clean_llm):
            return clean_llm
        
        # Generate simple rectified message
        return f"Fix {fix_type} issue in {row['File_Extension']} file"
    
    def is_precise_message(self, message, fix_type):
        message_lower = str(message).lower()
        return len(str(message).split()) > 3 and fix_type.lower() in message_lower
    
    def is_meaningful(self, message):
        stop_words = {'fix', 'update', 'change', 'modify', 'the', 'a', 'an'}
        words = set(str(message).lower().split()) - stop_words
        return len(words) > 2
    
    def analyze_rq1(self):
        """RQ1: Developer Precision Analysis"""
        print("RQ1 - DEVELOPER PRECISION ANALYSIS")
        print("-" * 50)
        
        # Calculate precision
        self.df['Dev_Precise'] = self.df.apply(
            lambda row: self.is_precise_message(row['Message'], row['Fix_Type']), axis=1
        )
        
        overall_rate = self.df['Dev_Precise'].mean()
        precise_count = self.df['Dev_Precise'].sum()
        total_count = len(self.df)
        
        print(f"Overall Hit Rate: {overall_rate:.2%} ({precise_count}/{total_count})")
        
        # By fix type analysis
        print(f"\nTable 1: Developer Precision by Fix Type")
        print("-" * 50)
        
        fix_results = []
        for fix_type in self.df['Fix_Type'].unique():
            subset = self.df[self.df['Fix_Type'] == fix_type]
            if len(subset) > 0:
                precise = subset['Dev_Precise'].sum()
                total = len(subset)
                rate = precise / total if total > 0 else 0
                fix_results.append({
                    'Fix_Type': fix_type,
                    'Total': total,
                    'Precise': precise,
                    'Hit_Rate': f"{rate:.3f}"
                })
        
        fix_df = pd.DataFrame(fix_results)
        print(fix_df.to_string(index=False))
        
        return overall_rate, precise_count, total_count, fix_df
    
    def analyze_rq2(self):
        """RQ2: LLM Precision Analysis"""
        print(f"\nRQ2 - LLM PRECISION ANALYSIS")
        print("-" * 50)
        
        # Define LLM precision
        self.df['LLM_Precise'] = (self.df['LLM_Length'] > 3) & (self.df['LLM_Clean'].apply(self.is_meaningful))
        
        overall_rate = self.df['LLM_Precise'].mean()
        precise_count = self.df['LLM_Precise'].sum()
        total_count = len(self.df)
        
        print(f"Overall Hit Rate: {overall_rate:.2%} ({precise_count}/{total_count})")
        
        # Developer vs LLM comparison
        dev_precise = self.df['Dev_Precise'].sum()
        llm_precise = self.df['LLM_Precise'].sum()
        both_precise = ((self.df['Dev_Precise']) & (self.df['LLM_Precise'])).sum()
        agreement = (self.df['Dev_Precise'] == self.df['LLM_Precise']).mean()
        
        print(f"\nComparison Summary:")
        print(f"Developer Precise: {dev_precise}/{total_count}")
        print(f"LLM Precise: {llm_precise}/{total_count}")
        print(f"Both Precise: {both_precise}/{total_count}")
        print(f"Agreement Rate: {agreement:.2%}")
        
        # Cross-tabulation table
        print(f"\nTable 2: Developer vs LLM Precision Cross-tabulation")
        print("-" * 60)
        
        dev_no_llm_no = ((~self.df['Dev_Precise']) & (~self.df['LLM_Precise'])).sum()
        dev_no_llm_yes = ((~self.df['Dev_Precise']) & (self.df['LLM_Precise'])).sum()
        dev_yes_llm_no = ((self.df['Dev_Precise']) & (~self.df['LLM_Precise'])).sum()
        dev_yes_llm_yes = ((self.df['Dev_Precise']) & (self.df['LLM_Precise'])).sum()
        
        crosstab_data = {
            '': ['Dev_Imprecise', 'Dev_Precise', 'Total'],
            'LLM_Imprecise': [dev_no_llm_no, dev_yes_llm_no, dev_no_llm_no + dev_yes_llm_no],
            'LLM_Precise': [dev_no_llm_yes, dev_yes_llm_yes, dev_no_llm_yes + dev_yes_llm_yes],
            'Total': [dev_no_llm_no + dev_no_llm_yes, dev_yes_llm_no + dev_yes_llm_yes, total_count]
        }
        
        crosstab_df = pd.DataFrame(crosstab_data)
        print(crosstab_df.to_string(index=False))
        
        return overall_rate, precise_count, total_count, agreement
    
    def analyze_rq3(self):
        """RQ3: Rectifier Effectiveness Analysis"""
        print(f"\nRQ3 - RECTIFIER EFFECTIVENESS ANALYSIS")
        print("-" * 50)
        
        # Simple improvement metric: message changed and got longer/better
        self.df['Improved'] = (
            (self.df['Rectified Message'] != self.df['Message']) &
            (self.df['Rectified_Length'] >= self.df['Message_Length'])
        )
        
        overall_rate = self.df['Improved'].mean()
        improved_count = self.df['Improved'].sum()
        total_count = len(self.df)
        
        print(f"Overall Improvement Rate: {overall_rate:.2%} ({improved_count}/{total_count})")
        
        # Average length changes
        avg_orig = self.df['Message_Length'].mean()
        avg_rect = self.df['Rectified_Length'].mean()
        
        print(f"Average Original Length: {avg_orig:.1f} words")
        print(f"Average Rectified Length: {avg_rect:.1f} words")
        print(f"Average Improvement: {avg_rect - avg_orig:.1f} words")
        
        # Improvement by fix type
        print(f"\nTable 3: Rectifier Improvement by Fix Type")
        print("-" * 50)
        
        improvement_results = []
        for fix_type in self.df['Fix_Type'].unique():
            subset = self.df[self.df['Fix_Type'] == fix_type]
            if len(subset) > 0:
                improved = subset['Improved'].sum()
                total = len(subset)
                rate = improved / total if total > 0 else 0
                improvement_results.append({
                    'Fix_Type': fix_type,
                    'Total': total,
                    'Improved': improved,
                    'Hit_Rate': f"{rate:.3f}"
                })
        
        improvement_df = pd.DataFrame(improvement_results)
        print(improvement_df.to_string(index=False))
        
        return overall_rate, improved_count, total_count, improvement_df
    
    def create_visualizations(self, rq1_rate, rq2_rate, rq3_rate):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Commit Message Rectification Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: RQ Comparison
        categories = ['RQ1: Developer\nPrecision', 'RQ2: LLM\nPrecision', 'RQ3: Rectifier\nImprovement']
        hit_rates = [rq1_rate, rq2_rate, rq3_rate]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = axes[0,0].bar(categories, hit_rates, color=colors, alpha=0.8)
        axes[0,0].set_ylabel('Hit Rate')
        axes[0,0].set_title('Research Questions Comparison', fontweight='bold')
        axes[0,0].set_ylim(0, 1)
        
        for bar, rate in zip(bars, hit_rates):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Fix Type Distribution
        fix_counts = self.df['Fix_Type'].value_counts().head(8)
        axes[0,1].pie(fix_counts.values, labels=fix_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Fix Type Distribution', fontweight='bold')
        
        # Plot 3: Developer Precision by Fix Type
        fix_precision = {}
        for fix_type in self.df['Fix_Type'].unique():
            subset = self.df[self.df['Fix_Type'] == fix_type]
            if len(subset) > 0:
                fix_precision[fix_type] = subset['Dev_Precise'].mean()
        
        sorted_precision = dict(sorted(fix_precision.items(), key=lambda x: x[1]))
        
        axes[0,2].barh(range(len(sorted_precision)), list(sorted_precision.values()), color='skyblue')
        axes[0,2].set_yticks(range(len(sorted_precision)))
        axes[0,2].set_yticklabels(list(sorted_precision.keys()))
        axes[0,2].set_xlabel('Hit Rate')
        axes[0,2].set_title('Developer Precision by Fix Type', fontweight='bold')
        
        # Plot 4: Message Length Comparison
        axes[1,0].hist([self.df['Message_Length'], self.df['Rectified_Length']], 
                       bins=15, alpha=0.7, label=['Original', 'Rectified'], color=['orange', 'green'])
        axes[1,0].set_xlabel('Message Length (words)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Message Length Distribution', fontweight='bold')
        axes[1,0].legend()
        
        # Plot 5: Precision Comparison Scatter
        dev_rates = []
        llm_rates = []
        labels = []
        
        for fix_type in self.df['Fix_Type'].unique():
            subset = self.df[self.df['Fix_Type'] == fix_type]
            if len(subset) > 2:  # Only include types with sufficient data
                dev_rates.append(subset['Dev_Precise'].mean())
                llm_rates.append(subset['LLM_Precise'].mean())
                labels.append(fix_type)
        
        if dev_rates and llm_rates:
            axes[1,1].scatter(dev_rates, llm_rates, s=100, alpha=0.7)
            axes[1,1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Agreement')
            axes[1,1].set_xlabel('Developer Precision Rate')
            axes[1,1].set_ylabel('LLM Precision Rate')
            axes[1,1].set_title('Developer vs LLM Precision', fontweight='bold')
            axes[1,1].legend()
        
        # Plot 6: Improvement by Fix Type
        improvement_rates = {}
        for fix_type in self.df['Fix_Type'].unique():
            subset = self.df[self.df['Fix_Type'] == fix_type]
            if len(subset) > 0:
                improvement_rates[fix_type] = subset['Improved'].mean()
        
        sorted_improvement = dict(sorted(improvement_rates.items(), key=lambda x: x[1]))
        
        axes[1,2].barh(range(len(sorted_improvement)), list(sorted_improvement.values()), color='lightgreen')
        axes[1,2].set_yticks(range(len(sorted_improvement)))
        axes[1,2].set_yticklabels(list(sorted_improvement.keys()))
        axes[1,2].set_xlabel('Improvement Rate')
        axes[1,2].set_title('Rectifier Improvement by Fix Type', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('commit_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate complete analysis report"""
        print("=" * 80)
        print("COMPREHENSIVE COMMIT MESSAGE ANALYSIS REPORT")
        print("=" * 80)
        
        # Analyze all research questions
        rq1_rate, rq1_precise, rq1_total, rq1_table = self.analyze_rq1()
        rq2_rate, rq2_precise, rq2_total, rq2_agreement = self.analyze_rq2()
        rq3_rate, rq3_improved, rq3_total, rq3_table = self.analyze_rq3()
        
        # Create visualizations
        self.create_visualizations(rq1_rate, rq2_rate, rq3_rate)
        
        # Generate executive summary
        print(f"\nEXECUTIVE SUMMARY")
        print("=" * 40)
        
        summary_data = {
            'Research Question': [
                'RQ1: Developer Precision',
                'RQ2: LLM Precision', 
                'RQ3: Rectifier Improvement'
            ],
            'Hit Rate': [
                f"{rq1_rate:.1%}",
                f"{rq2_rate:.1%}",
                f"{rq3_rate:.1%}"
            ],
            'Success Count': [
                f"{rq1_precise}/{rq1_total}",
                f"{rq2_precise}/{rq2_total}",
                f"{rq3_improved}/{rq3_total}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save results
        self.df.to_csv('final_analysis_results.csv', index=False)
        
        print(f"\nFiles Generated:")
        print("- final_analysis_results.csv: Complete dataset with rectified messages")
        print("- commit_analysis_results.png: All analysis visualizations")
        
        return self.df

# Usage
if __name__ == "__main__":
    analyzer = SimpleCommitAnalyzer('rectified_output.csv')
    results = analyzer.generate_summary_report()