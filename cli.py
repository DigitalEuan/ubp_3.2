"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Command Line Interface
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Simple CLI for running UBP scripts and operations.
"""

import sys
import os
import argparse
import json
from typing import Optional

from .dsl import eval_program, DSLParser, UBPParseError, UBPRuntimeError


def run_script(script_path: str, hardware_profile: str = "desktop_8gb", 
               output_file: Optional[str] = None, verbose: bool = False) -> int:
    """
    Run a UBP script file.
    
    Args:
        script_path: Path to .ubp script file
        hardware_profile: Hardware profile to use
        output_file: Optional output file for results
        verbose: Enable verbose output
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Read script file
        if not os.path.exists(script_path):
            print(f"Error: Script file '{script_path}' not found", file=sys.stderr)
            return 1
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        if verbose:
            print(f"Running UBP script: {script_path}")
            print(f"Hardware profile: {hardware_profile}")
        
        # Execute script
        results = eval_program(script_content, hardware_profile)
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"Results saved to: {output_file}")
        else:
            # Print summary to stdout
            if 'final_state' in results:
                runtime_state = results['final_state']['runtime_state']
                print(f"Simulation completed successfully")
                print(f"Final NRCI: {runtime_state.get('nrci_value', 'N/A')}")
                print(f"Total toggles: {runtime_state.get('total_toggles', 'N/A')}")
                print(f"Active realm: {runtime_state.get('active_realm', 'N/A')}")
                
                if 'performance_stats' in results['final_state']:
                    perf = results['final_state']['performance_stats']
                    print(f"Execution time: {perf.get('elapsed_time', 'N/A'):.4f}s")
            else:
                print("Script executed successfully")
        
        return 0
        
    except (UBPParseError, UBPRuntimeError) as e:
        print(f"UBP Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_interactive() -> int:
    """
    Run interactive UBP session.
    
    Returns:
        Exit code
    """
    print("UBP Interactive Session")
    print("Type 'help' for commands, 'exit' to quit")
    print()
    
    parser = DSLParser()
    
    while True:
        try:
            # Get user input
            line = input("ubp> ").strip()
            
            if not line:
                continue
            
            if line.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            
            if line.lower() == 'help':
                print_help()
                continue
            
            if line.lower() == 'reset':
                parser = DSLParser()
                print("Runtime reset")
                continue
            
            # Execute command
            try:
                results = parser.execute_script(line)
                
                # Print relevant results
                for key, value in results.items():
                    if isinstance(value, dict) and 'operation' in value:
                        print(f"Operation result: {value}")
                    elif key == 'final_state' and isinstance(value, dict):
                        runtime_state = value.get('runtime_state', {})
                        if 'nrci_value' in runtime_state:
                            print(f"NRCI: {runtime_state['nrci_value']:.6f}")
                        if 'total_toggles' in runtime_state:
                            print(f"Total toggles: {runtime_state['total_toggles']}")
                
            except (UBPParseError, UBPRuntimeError) as e:
                print(f"Error: {e}")
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            print("\nGoodbye!")
            break
    
    return 0


def print_help():
    """Print help information."""
    help_text = """
UBP Interactive Commands:

Basic Commands:
  init-runtime hardware=<profile>     Initialize runtime (desktop_8gb, mobile_4gb, raspberry_pi)
  use-realm <realm>                   Set active realm (quantum, electromagnetic, etc.)
  init-bitfield pattern=<p> density=<d> seed=<s>  Initialize Bitfield
  
Toggle Operations:
  toggle and <coord1> <coord2>        AND operation
  toggle xor <coord1> <coord2>        XOR operation  
  toggle or <coord1> <coord2>         OR operation
  toggle resonance <coord1> <coord2> frequency=<f> time=<t>  Resonance operation
  toggle entanglement <coord1> <coord2> coherence=<c>  Entanglement operation
  
Simulation:
  run-simulation steps=<n> ops_per_step=<m>  Run simulation
  get-metrics                         Get current metrics
  
Export:
  export-results <filename>           Export simulation results
  export-state <filename>             Export runtime state
  
Session:
  help                                Show this help
  reset                               Reset runtime
  exit                                Exit session

Coordinate format: [x,y,z,a,b,c] (6D coordinates)
Example: toggle xor [0,0,0,0,0,0] [0,0,0,0,0,1]
"""
    print(help_text)


def list_examples():
    """List available example scripts."""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
    
    if not os.path.exists(examples_dir):
        print("No examples directory found")
        return
    
    print("Available example scripts:")
    
    for filename in sorted(os.listdir(examples_dir)):
        if filename.endswith('.ubp'):
            filepath = os.path.join(examples_dir, filename)
            print(f"  {filename}")
            
            # Try to read first comment line as description
            try:
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#'):
                        description = first_line[1:].strip()
                        print(f"    {description}")
            except:
                pass
    
    print(f"\nRun with: ubp-run <script_name>")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UBP (Universal Binary Principle) Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ubp-run script.ubp                    Run a UBP script
  ubp-run script.ubp -o results.json   Run script and save results
  ubp-run -i                           Start interactive session
  ubp-run --list-examples               List available examples
        """
    )
    
    parser.add_argument('script', nargs='?', help='UBP script file to run')
    parser.add_argument('-i', '--interactive', action='store_true', 
                       help='Start interactive session')
    parser.add_argument('-o', '--output', help='Output file for results (JSON)')
    parser.add_argument('--hardware', default='desktop_8gb',
                       choices=['desktop_8gb', 'mobile_4gb', 'raspberry_pi'],
                       help='Hardware profile to use')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--list-examples', action='store_true',
                       help='List available example scripts')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.list_examples:
        list_examples()
        return 0
    
    if args.interactive:
        return run_interactive()
    
    if not args.script:
        parser.print_help()
        return 1
    
    # Run script
    return run_script(args.script, args.hardware, args.output, args.verbose)


if __name__ == '__main__':
    sys.exit(main())

