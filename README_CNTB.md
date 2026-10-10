# PSS/E Contingency Analysis (CNTB) Scripts

This repository contains Python scripts for performing contingency analysis in PSS/E using the CNTB module.

## Files Included

1. **`psse_contingency_analysis.py`** - Comprehensive class-based implementation
2. **`simple_cntb_example.py`** - Simple, straightforward example
3. **`requirements.txt`** - Python dependencies
4. **`README_CNTB.md`** - This documentation file

## Prerequisites

### Software Requirements
- **PSS/E** (Power System Simulator for Engineering) - Siemens PTI
- **Python** 3.7 or higher
- PSS/E Python API (`psspy` and `redirect` modules)

### Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

**Note:** The PSS/E Python API (`psspy` and `redirect`) must be installed as part of your PSS/E installation. These are proprietary components from Siemens PTI.

## Quick Start

### Using the Simple Example

1. **Update the case file path** in `simple_cntb_example.py`:
   ```python
   case_file = "your_case_file.sav"  # UPDATE THIS PATH
   ```

2. **Run the script**:
   ```bash
   python simple_cntb_example.py
   ```

### Using the Comprehensive Class

```python
from psse_contingency_analysis import PSSEContingencyAnalysis

# Initialize
cntb = PSSEContingencyAnalysis()

# Load your case
cntb.load_case("your_case_file.sav")

# Define contingencies
contingencies = [
    {
        'name': 'Line_1_2_Outage',
        'type': 'branch',
        'from_bus': 1,
        'to_bus': 2,
        'circuit': '1'
    },
    {
        'name': 'Generator_3_Outage',
        'type': 'generator',
        'bus': 3,
        'id': '1'
    }
]

# Create contingency file
cntb.create_contingency_file(contingencies)

# Setup options and run analysis
cntb.setup_cntb_options()
cntb.run_contingency_analysis()

# Export results
cntb.export_results_to_excel()
```

## Contingency Types Supported

### 1. Branch (Line) Outages
```python
{
    'name': 'Line_1_2_Outage',
    'type': 'branch',
    'from_bus': 1,
    'to_bus': 2,
    'circuit': '1'
}
```

### 2. Generator Outages
```python
{
    'name': 'Gen_3_Outage',
    'type': 'generator',
    'bus': 3,
    'id': '1'
}
```

### 3. Transformer Outages
```python
# Two-winding transformer
{
    'name': 'Transformer_1_3_Outage',
    'type': 'transformer',
    'from_bus': 1,
    'to_bus': 3,
    'circuit': '1'
}

# Three-winding transformer
{
    'name': 'Transformer_3W_Outage',
    'type': 'transformer',
    'from_bus': 1,
    'to_bus': 2,
    'winding3_bus': 3,
    'circuit': '1'
}
```

### 4. Load Outages
```python
{
    'name': 'Load_5_Outage',
    'type': 'load',
    'bus': 5,
    'id': '1'
}
```

## CNTB Options

You can customize the contingency analysis with various options:

```python
options = {
    'voltage_criteria': [0.95, 1.05],  # Min and max voltage p.u.
    'loading_criteria': 100.0,         # Max loading percentage
    'angle_criteria': 30.0,            # Max angle difference
    'iteration_limit': 10,             # Newton-Raphson iteration limit
    'acceleration_factor': 1.4,        # Acceleration factor
    'solution_tolerance': 0.0001       # Solution tolerance
}

cntb.setup_cntb_options(options)
```

## Output Files

The scripts generate several output files:

1. **`contingencies.con`** - PSS/E contingency definition file
2. **`cntb_results.txt`** - Raw CNTB output from PSS/E
3. **`contingency_summary.txt`** - Processed summary report
4. **`contingency_results.xlsx`** - Excel file with violation details

## Example Contingency File Format

```
/ PSS/E Contingency File
/ Generated automatically

CONTINGENCY Line_1_2_Outage
REMOVE BRANCH FROM BUS 1 TO BUS 2 CIRCUIT 1
END

CONTINGENCY Generator_3_Outage
REMOVE UNIT 1 FROM BUS 3
END

CONTINGENCY Transformer_1_4_Outage
REMOVE TRANSFORMER FROM BUS 1 TO BUS 4 CIRCUIT 1
END
```

## Error Handling

The scripts include comprehensive error handling for common issues:

- **Case file not found**
- **PSS/E initialization errors**
- **Base case solution failures**
- **CNTB execution errors**
- **Results parsing errors**

## Troubleshooting

### Common Issues

1. **PSS/E not found**
   - Ensure PSS/E is properly installed
   - Check that Python can import `psspy` and `redirect`

2. **Case file errors**
   - Verify the case file path is correct
   - Ensure the case file is a valid PSS/E .sav file

3. **Base case doesn't solve**
   - Check your power system model for errors
   - Verify generator and load data

4. **CNTB fails**
   - Check contingency definitions
   - Verify bus numbers and circuit IDs exist in your case

### Debug Mode

To enable more detailed output, you can modify the scripts to include debug information:

```python
# Add this for more detailed PSS/E output
psspy.report_output(2)  # Send output to screen
psspy.progress_output(2)  # Send progress to screen
```

## Advanced Usage

### Custom Contingency Analysis

```python
# Create custom contingencies programmatically
contingencies = []

# Add all line outages for a specific voltage level
for line in get_lines_by_voltage(138):  # Your function
    contingencies.append({
        'name': f'Line_{line.from_bus}_{line.to_bus}_Outage',
        'type': 'branch',
        'from_bus': line.from_bus,
        'to_bus': line.to_bus,
        'circuit': line.circuit
    })

# Run analysis
cntb.create_contingency_file(contingencies)
cntb.run_contingency_analysis()
```

### Batch Processing

```python
# Process multiple cases
case_files = ['case1.sav', 'case2.sav', 'case3.sav']

for case_file in case_files:
    cntb = PSSEContingencyAnalysis()
    cntb.load_case(case_file)
    cntb.create_contingency_file(standard_contingencies)
    cntb.run_contingency_analysis(f"results_{case_file}.txt")
```

## API Reference

### PSSEContingencyAnalysis Class

- `__init__(case_file=None)` - Initialize the class
- `load_case(case_file)` - Load PSS/E case file
- `create_contingency_file(contingencies, output_file)` - Create contingency file
- `setup_cntb_options(options)` - Configure CNTB options
- `run_contingency_analysis(output_file)` - Execute analysis
- `process_results(results_file)` - Process results
- `generate_summary_report(report_file)` - Create summary
- `export_results_to_excel(excel_file)` - Export to Excel

## Support

For issues related to:
- **PSS/E software**: Contact Siemens PTI support
- **These scripts**: Check the error messages and troubleshooting section
- **Power system modeling**: Consult PSS/E documentation

## License

These scripts are provided as-is for educational and professional use. PSS/E is a proprietary software product of Siemens PTI.