"""
PSS/E Contingency Analysis (CNTB) Script
=========================================

This script performs contingency analysis in PSS/E using the CNTB module.
It includes functions for setting up contingencies, running analysis, and processing results.

Author: Assistant
Date: 2024
"""

import os
import sys
import psspy
import redirect
import numpy as np
import pandas as pd
from datetime import datetime


class PSSEContingencyAnalysis:
    """
    A class to handle PSS/E Contingency Analysis operations
    """
    
    def __init__(self, case_file=None):
        """
        Initialize the contingency analysis class
        
        Args:
            case_file (str): Path to the PSS/E case file (.sav)
        """
        self.case_file = case_file
        self.contingency_file = None
        self.results = {}
        self.setup_psse()
    
    def setup_psse(self):
        """
        Initialize PSS/E environment
        """
        try:
            # Redirect PSS/E output
            redirect.psse2py()
            
            # Initialize PSS/E
            psspy.psseinit(150000)  # 150,000 bus limit
            
            print("PSS/E initialized successfully")
            
        except Exception as e:
            print(f"Error initializing PSS/E: {e}")
            sys.exit(1)
    
    def load_case(self, case_file=None):
        """
        Load PSS/E case file
        
        Args:
            case_file (str): Path to the case file
        """
        if case_file:
            self.case_file = case_file
        
        if not self.case_file or not os.path.exists(self.case_file):
            raise FileNotFoundError(f"Case file not found: {self.case_file}")
        
        try:
            ierr = psspy.case(self.case_file)
            if ierr != 0:
                raise Exception(f"Error loading case file: {ierr}")
            
            print(f"Case file loaded successfully: {self.case_file}")
            
        except Exception as e:
            print(f"Error loading case: {e}")
            raise
    
    def create_contingency_file(self, contingencies, output_file="contingencies.con"):
        """
        Create a contingency file for CNTB analysis
        
        Args:
            contingencies (list): List of contingency dictionaries
            output_file (str): Output contingency file name
            
        Example contingency format:
        [
            {
                'name': 'Line_1_2_Outage',
                'type': 'branch',
                'from_bus': 1,
                'to_bus': 2,
                'circuit': '1'
            },
            {
                'name': 'Gen_3_Outage',
                'type': 'generator',
                'bus': 3,
                'id': '1'
            }
        ]
        """
        self.contingency_file = output_file
        
        try:
            with open(output_file, 'w') as f:
                f.write("/ PSS/E Contingency File\n")
                f.write("/ Generated automatically\n")
                f.write("/ \n")
                
                for i, cont in enumerate(contingencies, 1):
                    f.write(f"CONTINGENCY {cont['name']}\n")
                    
                    if cont['type'].lower() == 'branch':
                        f.write(f"REMOVE BRANCH FROM BUS {cont['from_bus']} TO BUS {cont['to_bus']} CIRCUIT {cont['circuit']}\n")
                    
                    elif cont['type'].lower() == 'generator':
                        f.write(f"REMOVE UNIT {cont['id']} FROM BUS {cont['bus']}\n")
                    
                    elif cont['type'].lower() == 'load':
                        f.write(f"REMOVE LOAD {cont['id']} FROM BUS {cont['bus']}\n")
                    
                    elif cont['type'].lower() == 'transformer':
                        if 'winding3_bus' in cont:
                            # Three-winding transformer
                            f.write(f"REMOVE TRANSFORMER FROM BUS {cont['from_bus']} TO BUS {cont['to_bus']} TO BUS {cont['winding3_bus']} CIRCUIT {cont['circuit']}\n")
                        else:
                            # Two-winding transformer
                            f.write(f"REMOVE TRANSFORMER FROM BUS {cont['from_bus']} TO BUS {cont['to_bus']} CIRCUIT {cont['circuit']}\n")
                    
                    f.write("END\n")
                    f.write("\n")
            
            print(f"Contingency file created: {output_file}")
            
        except Exception as e:
            print(f"Error creating contingency file: {e}")
            raise
    
    def setup_cntb_options(self, options=None):
        """
        Set up CNTB analysis options
        
        Args:
            options (dict): Dictionary of CNTB options
        """
        default_options = {
            'voltage_criteria': [0.95, 1.05],  # Min and max voltage p.u.
            'loading_criteria': 100.0,         # Max loading percentage
            'angle_criteria': 30.0,            # Max angle difference
            'iteration_limit': 10,             # Newton-Raphson iteration limit
            'acceleration_factor': 1.4,        # Acceleration factor
            'solution_tolerance': 0.0001       # Solution tolerance
        }
        
        if options:
            default_options.update(options)
        
        self.options = default_options
        
        try:
            # Set CNTB options
            psspy.cntb_options(
                MAXITER=default_options['iteration_limit'],
                ACCEL=default_options['acceleration_factor'],
                TOLER=default_options['solution_tolerance']
            )
            
            print("CNTB options configured successfully")
            
        except Exception as e:
            print(f"Error setting CNTB options: {e}")
            raise
    
    def run_contingency_analysis(self, output_file="cntb_results.txt"):
        """
        Run the contingency analysis
        
        Args:
            output_file (str): Output file for results
        """
        if not self.contingency_file:
            raise ValueError("No contingency file specified")
        
        try:
            # Solve base case first
            ierr = psspy.fnsl()
            if ierr != 0:
                print(f"Warning: Base case solution error: {ierr}")
            
            # Run contingency analysis
            print("Starting contingency analysis...")
            
            ierr = psspy.cntb(
                STATUS=1,           # Status flag
                FILENAME=self.contingency_file,
                OUTFILE=output_file
            )
            
            if ierr != 0:
                raise Exception(f"CNTB analysis failed with error: {ierr}")
            
            print(f"Contingency analysis completed. Results saved to: {output_file}")
            
            # Process and store results
            self.process_results(output_file)
            
        except Exception as e:
            print(f"Error running contingency analysis: {e}")
            raise
    
    def process_results(self, results_file):
        """
        Process and analyze CNTB results
        
        Args:
            results_file (str): Path to CNTB results file
        """
        try:
            violations = {
                'voltage_violations': [],
                'loading_violations': [],
                'convergence_failures': []
            }
            
            # Read and parse results file
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    content = f.read()
                
                # Parse results (simplified parsing)
                lines = content.split('\n')
                current_contingency = None
                
                for line in lines:
                    line = line.strip()
                    
                    if 'CONTINGENCY' in line and 'RESULTS' in line:
                        current_contingency = line
                    
                    elif 'VOLTAGE VIOLATION' in line:
                        violations['voltage_violations'].append({
                            'contingency': current_contingency,
                            'details': line
                        })
                    
                    elif 'LOADING VIOLATION' in line or 'OVERLOAD' in line:
                        violations['loading_violations'].append({
                            'contingency': current_contingency,
                            'details': line
                        })
                    
                    elif 'CONVERGENCE FAILURE' in line or 'DID NOT CONVERGE' in line:
                        violations['convergence_failures'].append({
                            'contingency': current_contingency,
                            'details': line
                        })
            
            self.results = violations
            self.generate_summary_report()
            
        except Exception as e:
            print(f"Error processing results: {e}")
    
    def generate_summary_report(self, report_file="contingency_summary.txt"):
        """
        Generate a summary report of contingency analysis results
        
        Args:
            report_file (str): Output summary report file
        """
        try:
            with open(report_file, 'w') as f:
                f.write("PSS/E Contingency Analysis Summary Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Case File: {self.case_file}\n")
                f.write(f"Contingency File: {self.contingency_file}\n\n")
                
                # Summary statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Voltage Violations: {len(self.results.get('voltage_violations', []))}\n")
                f.write(f"Loading Violations: {len(self.results.get('loading_violations', []))}\n")
                f.write(f"Convergence Failures: {len(self.results.get('convergence_failures', []))}\n\n")
                
                # Detailed violations
                for violation_type, violations in self.results.items():
                    if violations:
                        f.write(f"{violation_type.upper().replace('_', ' ')}\n")
                        f.write("-" * 30 + "\n")
                        for i, violation in enumerate(violations, 1):
                            f.write(f"{i}. {violation.get('contingency', 'Unknown')}\n")
                            f.write(f"   Details: {violation.get('details', 'No details')}\n\n")
            
            print(f"Summary report generated: {report_file}")
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
    
    def get_bus_voltages_for_contingency(self, contingency_name):
        """
        Get bus voltages for a specific contingency
        
        Args:
            contingency_name (str): Name of the contingency
            
        Returns:
            dict: Bus voltages data
        """
        try:
            # This would require running the specific contingency
            # and extracting voltage data
            ierr, voltages = psspy.abusreal(-1, 2, "PU")
            ierr, bus_numbers = psspy.abusint(-1, 2, "NUMBER")
            
            if ierr == 0:
                voltage_data = {}
                for i, bus in enumerate(bus_numbers[0]):
                    voltage_data[bus] = voltages[0][i]
                
                return voltage_data
            
        except Exception as e:
            print(f"Error getting bus voltages: {e}")
            return {}
    
    def export_results_to_excel(self, excel_file="contingency_results.xlsx"):
        """
        Export results to Excel format
        
        Args:
            excel_file (str): Output Excel file name
        """
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                
                # Voltage violations
                if self.results.get('voltage_violations'):
                    voltage_df = pd.DataFrame(self.results['voltage_violations'])
                    voltage_df.to_excel(writer, sheet_name='Voltage_Violations', index=False)
                
                # Loading violations
                if self.results.get('loading_violations'):
                    loading_df = pd.DataFrame(self.results['loading_violations'])
                    loading_df.to_excel(writer, sheet_name='Loading_Violations', index=False)
                
                # Convergence failures
                if self.results.get('convergence_failures'):
                    convergence_df = pd.DataFrame(self.results['convergence_failures'])
                    convergence_df.to_excel(writer, sheet_name='Convergence_Failures', index=False)
            
            print(f"Results exported to Excel: {excel_file}")
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")


def main():
    """
    Main function demonstrating contingency analysis usage
    """
    try:
        # Initialize contingency analysis
        cntb_analysis = PSSEContingencyAnalysis()
        
        # Example case file (update path as needed)
        case_file = "example_case.sav"
        
        # Check if case file exists, if not create a simple example
        if not os.path.exists(case_file):
            print(f"Case file {case_file} not found. Please provide a valid PSS/E case file.")
            print("Example usage:")
            print("cntb_analysis.load_case('your_case_file.sav')")
            return
        
        # Load case
        cntb_analysis.load_case(case_file)
        
        # Define example contingencies
        example_contingencies = [
            {
                'name': 'Line_1_2_Outage',
                'type': 'branch',
                'from_bus': 1,
                'to_bus': 2,
                'circuit': '1'
            },
            {
                'name': 'Line_2_3_Outage',
                'type': 'branch',
                'from_bus': 2,
                'to_bus': 3,
                'circuit': '1'
            },
            {
                'name': 'Generator_1_Outage',
                'type': 'generator',
                'bus': 1,
                'id': '1'
            }
        ]
        
        # Create contingency file
        cntb_analysis.create_contingency_file(example_contingencies)
        
        # Setup CNTB options
        cntb_analysis.setup_cntb_options()
        
        # Run contingency analysis
        cntb_analysis.run_contingency_analysis()
        
        # Export results
        cntb_analysis.export_results_to_excel()
        
        print("Contingency analysis completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()