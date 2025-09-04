"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Domain-Specific Language (DSL) Parser
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Provides a simple scripting language for UBP operations:
- Bitfield initialization and manipulation
- Toggle operations and realm switching
- Simulation execution and result export
"""

import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from .runtime import Runtime, SimulationResult


@dataclass
class UBPCommand:
    """Represents a parsed UBP command."""
    command: str
    args: List[str]
    kwargs: Dict[str, Any]
    line_number: int


class UBPParseError(Exception):
    """Exception raised for UBP script parsing errors."""
    pass


class UBPRuntimeError(Exception):
    """Exception raised for UBP script runtime errors."""
    pass


class DSLParser:
    """
    UBP Domain-Specific Language Parser
    
    Parses and executes UBP scripts written in a simple command-based syntax.
    """
    
    def __init__(self):
        """Initialize the DSL parser."""
        self.runtime: Optional[Runtime] = None
        self.variables: Dict[str, Any] = {}
        self.commands: List[UBPCommand] = []
        
        # Command registry
        self.command_handlers = {
            'init-runtime': self._handle_init_runtime,
            'set-realm': self._handle_set_realm,
            'init-bitfield': self._handle_init_bitfield,
            'toggle': self._handle_toggle,
            'run-simulation': self._handle_run_simulation,
            'export-state': self._handle_export_state,
            'export-results': self._handle_export_results,
            'set-var': self._handle_set_var,
            'get-metrics': self._handle_get_metrics,
            'reset': self._handle_reset,
            'use-realm': self._handle_use_realm,
            'comment': self._handle_comment
        }
    
    def parse_script(self, script_content: str) -> List[UBPCommand]:
        """
        Parse a UBP script into commands.
        
        Args:
            script_content: Script content as string
            
        Returns:
            List of parsed commands
        """
        self.commands.clear()
        lines = script_content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            try:
                command = self._parse_line(line, line_num)
                if command:
                    self.commands.append(command)
            except Exception as e:
                raise UBPParseError(f"Line {line_num}: {str(e)}")
        
        return self.commands
    
    def _parse_line(self, line: str, line_number: int) -> Optional[UBPCommand]:
        """Parse a single line into a command."""
        # Handle parenthesized commands: (command arg1 arg2 key=value)
        if line.startswith('(') and line.endswith(')'):
            return self._parse_parenthesized_command(line[1:-1], line_number)
        
        # Handle simple commands: command arg1 arg2
        parts = line.split()
        if not parts:
            return None
        
        command = parts[0]
        args = []
        kwargs = {}
        
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                kwargs[key] = self._parse_value(value)
            else:
                args.append(self._parse_value(part))
        
        return UBPCommand(command, args, kwargs, line_number)
    
    def _parse_parenthesized_command(self, content: str, line_number: int) -> UBPCommand:
        """Parse a parenthesized command (Lisp-style syntax)."""
        # Simple tokenizer for parenthesized syntax
        tokens = self._tokenize(content)
        
        if not tokens:
            raise UBPParseError("Empty command")
        
        command = tokens[0]
        args = []
        kwargs = {}
        
        for token in tokens[1:]:
            if '=' in token:
                key, value = token.split('=', 1)
                kwargs[key] = self._parse_value(value)
            else:
                args.append(self._parse_value(token))
        
        return UBPCommand(command, args, kwargs, line_number)
    
    def _tokenize(self, content: str) -> List[str]:
        """Simple tokenizer for command parsing."""
        # Handle quoted strings and simple tokens
        tokens = []
        current_token = ""
        in_quotes = False
        
        for char in content:
            if char == '"' and not in_quotes:
                in_quotes = True
            elif char == '"' and in_quotes:
                in_quotes = False
                tokens.append(current_token)
                current_token = ""
            elif char == ' ' and not in_quotes:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value into appropriate Python type."""
        # Remove quotes if present
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Try to parse as boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
        
        # Try to parse as list
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value_str
    
    def execute_script(self, script_content: str) -> Dict[str, Any]:
        """
        Execute a UBP script.
        
        Args:
            script_content: Script content as string
            
        Returns:
            Execution results and final state
        """
        # Parse script
        commands = self.parse_script(script_content)
        
        # Execute commands
        results = {}
        
        for command in commands:
            try:
                result = self._execute_command(command)
                if result is not None:
                    results[f"line_{command.line_number}"] = result
            except Exception as e:
                raise UBPRuntimeError(f"Line {command.line_number}: {str(e)}")
        
        # Add final runtime state
        if self.runtime:
            results['final_state'] = {
                'runtime_state': self.runtime.state.to_dict(),
                'performance_stats': self.runtime.get_performance_stats()
            }
        
        return results
    
    def _execute_command(self, command: UBPCommand) -> Any:
        """Execute a single command."""
        handler = self.command_handlers.get(command.command)
        if not handler:
            raise UBPRuntimeError(f"Unknown command: {command.command}")
        
        return handler(command.args, command.kwargs)
    
    # Command handlers
    
    def _handle_init_runtime(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Initialize the UBP runtime."""
        hardware_profile = kwargs.get('hardware', 'desktop_8gb')
        self.runtime = Runtime(hardware_profile)
        return f"Runtime initialized with {hardware_profile} profile"
    
    def _handle_set_realm(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Set the active realm."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        if not args:
            raise UBPRuntimeError("Realm name required")
        
        realm_name = args[0]
        self.runtime.set_realm(realm_name)
        return f"Active realm set to {realm_name}"
    
    def _handle_use_realm(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Alias for set-realm command."""
        return self._handle_set_realm(args, kwargs)
    
    def _handle_init_bitfield(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Initialize the Bitfield."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        pattern = kwargs.get('pattern', 'sparse_random')
        density = kwargs.get('density', 0.01)
        seed = kwargs.get('seed', None)
        
        self.runtime.initialize_bitfield(pattern, density, seed)
        return f"Bitfield initialized with pattern={pattern}, density={density}"
    
    def _handle_toggle(self, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a toggle operation."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        if len(args) < 3:
            raise UBPRuntimeError("Toggle requires: operation coord1 coord2")
        
        operation = args[0]
        coord1 = tuple(args[1]) if isinstance(args[1], list) else (0, 0, 0, 0, 0, 0)
        coord2 = tuple(args[2]) if isinstance(args[2], list) else (0, 0, 0, 0, 0, 1)
        
        result = self.runtime.execute_toggle_operation(operation, coord1, coord2, **kwargs)
        
        return {
            'operation': operation,
            'result_value': result.value,
            'result_layers': {
                'reality': result.reality_layer,
                'information': result.information_layer,
                'activation': result.activation_layer,
                'unactivated': result.unactivated_layer
            }
        }
    
    def _handle_run_simulation(self, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Run a UBP simulation."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        steps = kwargs.get('steps', 100)
        operations_per_step = kwargs.get('ops_per_step', 10)
        record_timeline = kwargs.get('timeline', True)
        
        result = self.runtime.run_simulation(
            steps=steps,
            operations_per_step=operations_per_step,
            record_timeline=record_timeline
        )
        
        # Store result for potential export
        self.variables['last_simulation'] = result
        
        return {
            'steps_completed': steps,
            'execution_time': result.execution_time,
            'final_metrics': result.metrics,
            'nrci': result.final_state.nrci_value
        }
    
    def _handle_export_state(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Export runtime state to file."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        if not args:
            raise UBPRuntimeError("Filepath required")
        
        filepath = args[0]
        format_type = kwargs.get('format', 'json')
        
        self.runtime.export_state(filepath, format_type)
        return f"State exported to {filepath}"
    
    def _handle_export_results(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Export simulation results to file."""
        if 'last_simulation' not in self.variables:
            raise UBPRuntimeError("No simulation results to export")
        
        if not args:
            raise UBPRuntimeError("Filepath required")
        
        filepath = args[0]
        result: SimulationResult = self.variables['last_simulation']
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return f"Results exported to {filepath}"
    
    def _handle_set_var(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Set a variable."""
        if len(args) < 2:
            raise UBPRuntimeError("set-var requires: name value")
        
        name = args[0]
        value = args[1]
        self.variables[name] = value
        
        return f"Variable {name} set to {value}"
    
    def _handle_get_metrics(self, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get current runtime metrics."""
        if not self.runtime:
            raise UBPRuntimeError("Runtime not initialized")
        
        return {
            'bitfield_stats': {
                'active_count': self.runtime.bitfield.active_count,
                'total_offbits': self.runtime.bitfield.total_offbits,
                'sparsity': self.runtime.bitfield.current_sparsity,
                'toggle_count': self.runtime.bitfield.toggle_count
            },
            'runtime_state': self.runtime.state.to_dict(),
            'performance': self.runtime.get_performance_stats()
        }
    
    def _handle_reset(self, args: List[Any], kwargs: Dict[str, Any]) -> str:
        """Reset the runtime."""
        if self.runtime:
            self.runtime.reset()
        self.variables.clear()
        return "Runtime reset"
    
    def _handle_comment(self, args: List[Any], kwargs: Dict[str, Any]) -> None:
        """Handle comment (no-op)."""
        return None


def parse_ubp_script(script_content: str) -> List[UBPCommand]:
    """
    Parse a UBP script into commands.
    
    Args:
        script_content: Script content as string
        
    Returns:
        List of parsed commands
    """
    parser = DSLParser()
    return parser.parse_script(script_content)


def eval_program(script_content: str, hardware_profile: str = "desktop_8gb") -> Dict[str, Any]:
    """
    Evaluate a complete UBP program.
    
    Args:
        script_content: UBP script content
        hardware_profile: Hardware profile to use
        
    Returns:
        Program execution results
    """
    parser = DSLParser()
    
    # Auto-initialize runtime if not done in script
    if 'init-runtime' not in script_content:
        init_script = f"init-runtime hardware={hardware_profile}\n"
        script_content = init_script + script_content
    
    return parser.execute_script(script_content)


# Example UBP script templates

QUANTUM_SIMULATION_TEMPLATE = '''
# UBP Quantum Realm Simulation
init-runtime hardware=desktop_8gb
set-realm quantum
init-bitfield pattern=quantum_bias density=0.01 seed=42

# Run simulation
run-simulation steps=100 ops_per_step=10 timeline=true

# Export results
export-results quantum_simulation_results.json
get-metrics
'''

MULTI_REALM_TEMPLATE = '''
# UBP Multi-Realm Simulation
init-runtime hardware=desktop_8gb

# Initialize with quantum bias
use-realm quantum
init-bitfield pattern=quantum_bias density=0.01

# Run quantum phase
run-simulation steps=50 ops_per_step=5
export-results quantum_phase.json

# Switch to electromagnetic realm
use-realm electromagnetic
run-simulation steps=50 ops_per_step=5
export-results electromagnetic_phase.json

# Final metrics
get-metrics
'''

TOGGLE_OPERATIONS_TEMPLATE = '''
# UBP Toggle Operations Demo
init-runtime
init-bitfield pattern=sparse_random density=0.005

# Execute various toggle operations
toggle xor [0,0,0,0,0,0] [0,0,0,0,0,1]
toggle resonance [1,1,1,0,0,0] [1,1,1,0,0,1] frequency=1000.0
toggle entanglement [2,2,2,0,0,0] [2,2,2,0,0,1] coherence=0.95

# Run short simulation
run-simulation steps=10 ops_per_step=3
get-metrics
'''

