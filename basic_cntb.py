"""
Basic PSS/E CNTB Example - Minimal Code
=======================================

This script shows the absolute minimum code needed to run CNTB in PSS/E.
"""

import psspy
import redirect

# Initialize PSS/E
redirect.psse2py()
psspy.psseinit(50000)

# Load your case file
psspy.case("your_case_file.sav")  # Replace with your case file path

# Solve base case
psspy.fnsl()

# Create a simple contingency file
with open("basic_contingencies.con", "w") as f:
    f.write("""
CONTINGENCY Line_Outage_1
REMOVE BRANCH FROM BUS 1 TO BUS 2 CIRCUIT 1
END

CONTINGENCY Gen_Outage_1
REMOVE UNIT 1 FROM BUS 1
END
""")

# Run contingency analysis
psspy.cntb(
    STATUS=1,                           # Run analysis
    FILENAME="basic_contingencies.con", # Contingency file
    OUTFILE="basic_results.txt"         # Output file
)

print("CNTB analysis completed. Check basic_results.txt for results.")

# Optional: Display some results
try:
    with open("basic_results.txt", "r") as f:
        print("\n--- First 1000 characters of results ---")
        print(f.read()[:1000])
except:
    print("Could not read results file.")