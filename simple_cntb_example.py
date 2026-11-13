"""
Simple PSS/E Contingency Analysis (CNTB) Example
================================================

This is a simplified example of how to perform contingency analysis in PSS/E.
Use this as a starting point for your own contingency analysis scripts.

Requirements:
- PSS/E installation with Python API
- Valid PSS/E case file (.sav)
"""

import psspy
import redirect
import os


def initialize_psse():
    """Initialize PSS/E"""
    try:
        redirect.psse2py()
        psspy.psseinit(50000)  # 50,000 bus limit
        print("PSS/E initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing PSS/E: {e}")
        return False


def load_case(case_file):
    """Load PSS/E case file"""
    try:
        ierr = psspy.case(case_file)
        if ierr == 0:
            print(f"Case loaded: {case_file}")
            return True
        else:
            print(f"Error loading case: {ierr}")
            return False
    except Exception as e:
        print(f"Exception loading case: {e}")
        return False


def solve_base_case():
    """Solve the base case"""
    try:
        ierr = psspy.fnsl()
        if ierr == 0:
            print("Base case solved successfully")
            return True
        else:
            print(f"Base case solution error: {ierr}")
            return False
    except Exception as e:
        print(f"Exception solving base case: {e}")
        return False


def create_simple_contingency_file():
    """Create a simple contingency file"""
    contingency_content = """
/ Simple Contingency File for PSS/E CNTB
/ Line outage contingencies

CONTINGENCY Line_Outage_Example
REMOVE BRANCH FROM BUS 1 TO BUS 2 CIRCUIT 1
END

CONTINGENCY Generator_Outage_Example  
REMOVE UNIT 1 FROM BUS 1
END

CONTINGENCY Transformer_Outage_Example
REMOVE TRANSFORMER FROM BUS 1 TO BUS 3 CIRCUIT 1
END
"""
    
    try:
        with open("simple_contingencies.con", "w") as f:
            f.write(contingency_content)
        print("Contingency file created: simple_contingencies.con")
        return "simple_contingencies.con"
    except Exception as e:
        print(f"Error creating contingency file: {e}")
        return None


def run_contingency_analysis(contingency_file, output_file="cntb_output.txt"):
    """Run the contingency analysis"""
    try:
        print("Starting contingency analysis...")
        
        # Run CNTB
        ierr = psspy.cntb(
            STATUS=1,                    # Status flag (1 = run analysis)
            FILENAME=contingency_file,   # Contingency file
            OUTFILE=output_file         # Output file
        )
        
        if ierr == 0:
            print(f"Contingency analysis completed successfully!")
            print(f"Results saved to: {output_file}")
            return True
        else:
            print(f"CNTB failed with error code: {ierr}")
            return False
            
    except Exception as e:
        print(f"Exception during contingency analysis: {e}")
        return False


def read_results(output_file):
    """Read and display basic results"""
    try:
        if os.path.exists(output_file):
            print(f"\n--- Contents of {output_file} ---")
            with open(output_file, 'r') as f:
                content = f.read()
                print(content[:2000])  # Show first 2000 characters
                if len(content) > 2000:
                    print("\n... (output truncated)")
        else:
            print(f"Output file {output_file} not found")
    except Exception as e:
        print(f"Error reading results: {e}")


def main():
    """Main function"""
    print("PSS/E Contingency Analysis Example")
    print("=" * 40)
    
    # Step 1: Initialize PSS/E
    if not initialize_psse():
        return
    
    # Step 2: Load case file (update this path to your case file)
    case_file = "your_case_file.sav"  # UPDATE THIS PATH
    
    if not os.path.exists(case_file):
        print(f"\nWarning: Case file '{case_file}' not found.")
        print("Please update the 'case_file' variable with the path to your PSS/E case file.")
        print("\nFor demonstration, creating a simple contingency file anyway...")
        create_simple_contingency_file()
        return
    
    if not load_case(case_file):
        return
    
    # Step 3: Solve base case
    if not solve_base_case():
        return
    
    # Step 4: Create contingency file
    contingency_file = create_simple_contingency_file()
    if not contingency_file:
        return
    
    # Step 5: Run contingency analysis
    output_file = "cntb_results.txt"
    if run_contingency_analysis(contingency_file, output_file):
        # Step 6: Display results
        read_results(output_file)
    
    print("\nContingency analysis example completed!")


# Alternative function for custom contingencies
def run_custom_contingencies(case_file, contingencies_list):
    """
    Run contingency analysis with custom contingencies
    
    Args:
        case_file (str): Path to PSS/E case file
        contingencies_list (list): List of contingency definitions
        
    Example:
        contingencies = [
            "CONTINGENCY Line_1_2\nREMOVE BRANCH FROM BUS 1 TO BUS 2 CIRCUIT 1\nEND\n",
            "CONTINGENCY Gen_3\nREMOVE UNIT 1 FROM BUS 3\nEND\n"
        ]
    """
    
    # Initialize and load case
    if not initialize_psse() or not load_case(case_file) or not solve_base_case():
        return False
    
    # Create custom contingency file
    contingency_file = "custom_contingencies.con"
    try:
        with open(contingency_file, 'w') as f:
            f.write("/ Custom Contingency File\n")
            for contingency in contingencies_list:
                f.write(contingency + "\n")
        
        print(f"Custom contingency file created: {contingency_file}")
        
        # Run analysis
        return run_contingency_analysis(contingency_file, "custom_cntb_results.txt")
        
    except Exception as e:
        print(f"Error creating custom contingency file: {e}")
        return False


if __name__ == "__main__":
    main()