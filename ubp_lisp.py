"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP-Lisp: Native Computational Ontology and BitBase for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

Implements the complete UBP-Lisp language and BitBase system that serves
as the native computational ontology for the UBP framework. Provides
domain-specific language constructs for UBP operations, BitBase storage,
and JIT compilation capabilities.

Mathematical Foundation:
- S-expression based syntax for UBP operations
- BitBase: Content-addressable storage for UBP computations (now leveraging HexDictionary)
- Native UBP primitives: toggle, resonance, entanglement, etc.
- JIT compilation for performance optimization
- Ontological type system for UBP entities

"""

import numpy as np
import math
import ast
import hashlib
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import re

# Import HexDictionary for persistent storage
from hex_dictionary import HexDictionary
# Import UBPConfig for constants
from ubp_config import get_config, UBPConfig
from state import OffBit # Needed for some UBP operations

_config: UBPConfig = get_config() # Initialize configuration

class UBPType(Enum):
    """UBP-Lisp data types"""
    OFFBIT = "offbit"                  # 24-bit OffBit
    BITFIELD = "bitfield"             # 6D Bitfield
    REALM = "realm"                   # UBP realm
    FREQUENCY = "frequency"           # Resonance frequency
    COHERENCE = "coherence"           # Coherence value
    TENSOR = "tensor"                 # Purpose tensor
    GLYPH = "glyph"                   # Rune Protocol glyph
    FUNCTION = "function"             # UBP-Lisp function
    SYMBOL = "symbol"                 # Lisp symbol
    NUMBER = "number"                 # Numeric value
    LIST = "list"                     # Lisp list
    STRING = "string"                 # String literal
    BOOLEAN = "boolean"               # Boolean value
    NIL = "nil"                       # Nil value


@dataclass
class UBPValue:
    """
    Represents a value in UBP-Lisp with type information.
    """
    value: Any
    ubp_type: UBPType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        return f"{self.ubp_type.value}:{self.value}"


@dataclass
class BitBaseEntry: # Still used internally to represent retrieved data from HexDictionary
    """
    Entry in the BitBase content-addressable storage.
    """
    content_hash: str
    content: Any
    ubp_type: UBPType
    timestamp: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BitBase:
    """
    Content-addressable storage system for UBP computations.
    Now acts as a wrapper around the persistent HexDictionary.
    """
    
    def __init__(self, hex_dict_instance: Optional[HexDictionary] = None):
        self.hex_dict = hex_dict_instance if hex_dict_instance else HexDictionary()
        self.statistics = { # These statistics now track interaction with the underlying HexDictionary
            'total_stores': 0,
            'total_retrievals': 0,
            'cache_hits': 0, # Placeholder, as HexDict handles its own hits
            'cache_misses': 0 # Placeholder
        }
        print("UBP-Lisp BitBase initialized using HexDictionary for persistent storage.")

    def store(self, content: Any, ubp_type: UBPType, 
             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store content in HexDictionary and return content hash.
        """
        # HexDictionary handles serialization and hashing internally.
        # It also automatically creates a content hash.
        
        # Augment metadata for HexDictionary, including UBPType
        full_metadata = {
            "ubp_lisp_type": ubp_type.value,
            "original_lisp_metadata": metadata if metadata is not None else {}
        }
        
        # HexDictionary's store method needs a simple type string, not UBPType enum directly.
        # We also ensure content is serializable.
        hex_dict_data_type = self._map_ubp_type_to_hex_dict_type(ubp_type, content)
        
        content_hash = self.hex_dict.store(content, hex_dict_data_type, metadata=full_metadata)
        
        self.statistics['total_stores'] += 1
        return content_hash
    
    def retrieve(self, content_hash: str) -> Optional[BitBaseEntry]:
        """
        Retrieve content from HexDictionary by hash.
        """
        self.statistics['total_retrievals'] += 1
        
        retrieved_data = self.hex_dict.retrieve(content_hash)
        retrieved_meta = self.hex_dict.get_metadata(content_hash)
        
        if retrieved_data is None or retrieved_meta is None:
            return None
        
        # Reconstruct UBPValue from HexDictionary entry
        ubp_lisp_type_str = retrieved_meta.get("ubp_lisp_type")
        original_lisp_metadata = retrieved_meta.get("original_lisp_metadata", {})
        
        ubp_type = UBPType(ubp_lisp_type_str) if ubp_lisp_type_str else UBPType.NIL
        
        # HexDictionary internally tracks access; for this wrapper, we just assume a "hit" if retrieved
        self.statistics['cache_hits'] += 1 
        
        return BitBaseEntry(
            content_hash=content_hash,
            content=retrieved_data,
            ubp_type=ubp_type,
            timestamp=time.time(), # This would be retrieval time, not original store time
            access_count=retrieved_meta.get('access_count', 0) + 1, # HexDict tracks this
            metadata=original_lisp_metadata
        )
    
    def _map_ubp_type_to_hex_dict_type(self, ubp_type: UBPType, content: Any) -> str:
        """Maps UBPType to HexDictionary's simple type strings."""
        if ubp_type == UBPType.NUMBER:
            return 'int' if isinstance(content, int) else 'float'
        elif ubp_type == UBPType.STRING:
            return 'str'
        elif ubp_type == UBPType.BOOLEAN:
            return 'str' # HexDictionary doesn't have a specific bool type, store as str
        elif ubp_type == UBPType.LIST or ubp_type == UBPType.BITFIELD: # Assuming Bitfield stores as a list/array
            if isinstance(content, np.ndarray):
                return 'array'
            return 'list'
        elif ubp_type == UBPType.OFFBIT:
            return 'int' # OffBit is an integer value
        elif ubp_type == UBPType.TENSOR:
            return 'array' # Assume tensors are numpy arrays
        elif ubp_type == UBPType.FUNCTION:
            return 'json' # Store function definition as JSON/string
        else:
            return 'json' # Default for complex types

    def get_statistics(self) -> Dict[str, Any]:
        """Get BitBase statistics, deferring to HexDictionary for details."""
        hd_stats = self.hex_dict.get_metadata_stats() # Assuming HexDictionary has a get_metadata_stats()
        
        return {
            'total_entries': len(self.hex_dict),
            'max_entries': 'N/A (HexDictionary dynamic)', # HexDictionary doesn't have a fixed max_entries in this sense
            'total_stores': self.statistics['total_stores'],
            'total_retrievals': self.statistics['total_retrievals'],
            'hex_dict_cache_hit_rate': hd_stats.get('cache_hit_rate', 'N/A'), # Use HexDict's actual hit rate
            'hex_dict_storage_usage_bytes': hd_stats.get('storage_size_bytes', 'N/A')
        }


class UBPLispEnvironment:
    """
    Environment for UBP-Lisp variable bindings and function definitions.
    """
    
    def __init__(self, parent: Optional['UBPLispEnvironment'] = None):
        self.parent = parent
        self.bindings = {}
        self.functions = {}
    
    def define(self, symbol: str, value: UBPValue):
        """Define a variable in this environment"""
        self.bindings[symbol] = value
    
    def lookup(self, symbol: str) -> Optional[UBPValue]:
        """Look up a variable in this environment or parent environments"""
        if symbol in self.bindings:
            return self.bindings[symbol]
        elif self.parent:
            return self.parent.lookup(symbol)
        else:
            return None
    
    def define_function(self, name: str, params: List[str], body: Any):
        """Define a function in this environment"""
        self.functions[name] = {
            'params': params,
            'body': body,
            'closure': self
        }
    
    def lookup_function(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a function in this environment or parent environments"""
        if name in self.functions:
            return self.functions[name]
        elif self.parent:
            return self.parent.lookup_function(name)
        else:
            return None


class UBPLispParser:
    """
    Parser for UBP-Lisp S-expressions with UBP-specific syntax extensions.
    """
    
    def __init__(self):
        self.token_patterns = [
            (r'\s+', None),                    # Whitespace (ignore)
            (r';[^\n]*', None),                # Comments (ignore)
            (r'\(', 'LPAREN'),                 # Left parenthesis
            (r'\)', 'RPAREN'),                 # Right parenthesis
            (r'\[', 'LBRACKET'),               # Left bracket
            (r'\]', 'RBRACKET'),               # Right bracket
            (r'"[^"]*"', 'STRING'),            # String literal
            (r'#[tf]', 'BOOLEAN'),             # Boolean literal
            (r'#b[01]+', 'BINARY'),            # Binary literal
            (r'#x[0-9a-fA-F]+', 'HEX'),       # Hexadecimal literal
            (r'-?\d+\.\d+', 'FLOAT'),          # Float literal
            (r'-?\d+', 'INTEGER'),             # Integer literal
            (r'[a-zA-Z_+\-*/=<>][a-zA-Z0-9_\-\*\+\?=<>]*', 'SYMBOL'),  # Symbol (including operators)
            (r'\.', 'DOT'),                    # Dot
            (r"'", 'QUOTE'),                   # Quote
        ]
        
        self.compiled_patterns = [(re.compile(pattern), token_type) 
                                 for pattern, token_type in self.token_patterns]
    
    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize UBP-Lisp source code.
        
        Args:
            text: Source code text
        
        Returns:
            List of (token_value, token_type) tuples
        """
        tokens = []
        position = 0
        
        while position < len(text):
            matched = False
            
            for pattern, token_type in self.compiled_patterns:
                match = pattern.match(text, position)
                if match:
                    token_value = match.group(0)
                    if token_type is not None:  # Skip whitespace and comments
                        tokens.append((token_value, token_type))
                    position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Unexpected character at position {position}: {text[position]}")
        
        return tokens
    
    def parse(self, tokens: List[Tuple[str, str]]) -> Any:
        """
        Parse tokens into UBP-Lisp AST.
        
        Args:
            tokens: List of tokens from tokenizer
        
        Returns:
            Parsed AST
        """
        self.tokens = tokens
        self.position = 0
        
        if not tokens:
            return None
        
        return self._parse_expression()
    
    def _parse_expression(self) -> Any:
        """Parse a single expression"""
        if self.position >= len(self.tokens):
            raise SyntaxError("Unexpected end of input")
        
        token_value, token_type = self.tokens[self.position]
        
        if token_type == 'LPAREN':
            return self._parse_list()
        elif token_type == 'LBRACKET':
            return self._parse_vector()
        elif token_type == 'QUOTE':
            self.position += 1
            return ['quote', self._parse_expression()]
        elif token_type == 'INTEGER':
            self.position += 1
            return int(token_value)
        elif token_type == 'FLOAT':
            self.position += 1
            return float(token_value)
        elif token_type == 'STRING':
            self.position += 1
            return token_value[1:-1]  # Remove quotes
        elif token_type == 'BOOLEAN':
            self.position += 1
            return token_value == '#t'
        elif token_type == 'BINARY':
            self.position += 1
            return int(token_value[2:], 2)  # Remove #b prefix
        elif token_type == 'HEX':
            self.position += 1
            return int(token_value[2:], 16)  # Remove #x prefix
        elif token_type == 'SYMBOL':
            self.position += 1
            return token_value
        else:
            raise SyntaxError(f"Unexpected token: {token_value}")
    
    def _parse_list(self) -> List[Any]:
        """Parse a list expression"""
        self.position += 1  # Skip opening paren
        elements = []
        
        while self.position < len(self.tokens):
            token_value, token_type = self.tokens[self.position]
            
            if token_type == 'RPAREN':
                self.position += 1
                return elements
            else:
                elements.append(self._parse_expression())
        
        raise SyntaxError("Unclosed list")
    
    def _parse_vector(self) -> List[Any]:
        """Parse a vector expression"""
        self.position += 1  # Skip opening bracket
        elements = []
        
        while self.position < len(self.tokens):
            token_value, token_type = self.tokens[self.position]
            
            if token_type == 'RBRACKET':
                self.position += 1
                return ['vector'] + elements
            else:
                elements.append(self._parse_expression())
        
        raise SyntaxError("Unclosed vector")


class UBPLispInterpreter:
    """
    Main UBP-Lisp interpreter with native UBP operations.
    
    Provides evaluation of UBP-Lisp expressions with built-in UBP primitives
    and integration with the BitBase storage system.
    """
    
    def __init__(self, hex_dict_instance: Optional[HexDictionary] = None):
        self.parser = UBPLispParser()
        self.bitbase = BitBase(hex_dict_instance=hex_dict_instance) # Inject HexDictionary
        self.global_env = UBPLispEnvironment()
        self.call_stack = []
        
        # Initialize built-in functions
        self._initialize_builtins()
    
    def _initialize_builtins(self):
        """Initialize built-in UBP-Lisp functions"""
        
        print("DEBUG(UBP-Lisp): Initializing built-in functions...")
        
        # Arithmetic operations
        self.global_env.define_function('+', ['&rest', 'args'], self._builtin_add)
        self.global_env.define_function('-', ['&rest', 'args'], self._builtin_subtract)
        self.global_env.define_function('*', ['&rest', 'args'], self._builtin_multiply)
        self.global_env.define_function('/', ['&rest', 'args'], self._builtin_divide)
        
        # Comparison operations
        self.global_env.define_function('=', ['a', 'b'], self._builtin_equal)
        self.global_env.define_function('<', ['a', 'b'], self._builtin_less)
        self.global_env.define_function('>', ['a', 'b'], self._builtin_greater)
        self.global_env.define_function('<=', ['a', 'b'], self._builtin_less_equal)
        self.global_env.define_function('>=', ['a', 'b'], self._builtin_greater_equal)
        
        # List operations
        self.global_env.define_function('cons', ['a', 'b'], self._builtin_cons)
        self.global_env.define_function('car', ['list'], self._builtin_car)
        self.global_env.define_function('cdr', ['list'], self._builtin_cdr)
        self.global_env.define_function('list', ['&rest', 'args'], self._builtin_list)
        self.global_env.define_function('length', ['list'], self._builtin_length)
        
        # UBP-specific operations
        self.global_env.define_function('make-offbit', ['value'], self._builtin_make_offbit)
        self.global_env.define_function('toggle', ['offbit'], self._builtin_toggle)
        self.global_env.define_function('resonance', ['freq1', 'freq2'], self._builtin_resonance)
        self.global_env.define_function('entangle', ['bit1', 'bit2'], self._builtin_entangle)
        self.global_env.define_function('coherence', ['bitfield_or_values'], self._builtin_coherence)
        self.global_env.define_function('spin-transition', ['bit', 'realm'], self._builtin_spin_transition)
        
        # BitBase operations
        self.global_env.define_function('store', ['value'], self._builtin_store)
        self.global_env.define_function('retrieve', ['hash'], self._builtin_retrieve)
        
        # Control flow
        self.global_env.define_function('if', ['condition', 'then', 'else'], self._evaluate_if) # Changed to call directly
        self.global_env.define_function('cond', ['&rest', 'clauses'], self._builtin_cond)
        
        # Function definition
        self.global_env.define_function('defun', ['name', 'params', 'body'], self._evaluate_defun) # Changed to call directly
        self.global_env.define_function('lambda', ['params', 'body'], self._evaluate_lambda) # Changed to call directly
        
        # Variable definition
        self.global_env.define_function('define', ['symbol', 'value'], self._evaluate_define) # Changed to call directly
        self.global_env.define_function('let', ['bindings', 'body'], self._evaluate_let) # Changed to call directly
        
        print(f"DEBUG(UBP-Lisp): Built-in functions defined: {list(self.global_env.functions.keys())}")
        print(f"DEBUG(UBP-Lisp): Type of self._builtin_equal: {type(self._builtin_equal)}")
    
    def evaluate(self, expression: Any, env: Optional[UBPLispEnvironment] = None) -> UBPValue:
        """
        Evaluate a UBP-Lisp expression.
        
        Args:
            expression: Expression to evaluate
            env: Environment for evaluation (uses global if None)
        
        Returns:
            Evaluated UBP value
        """
        if env is None:
            env = self.global_env
        
        # Self-evaluating expressions
        if isinstance(expression, (int, float)):
            return UBPValue(expression, UBPType.NUMBER)
        elif isinstance(expression, bool):
            return UBPValue(expression, UBPType.BOOLEAN)
        elif expression is None:
            return UBPValue(None, UBPType.NIL)
        
        # Symbol lookup (for variable names)
        elif isinstance(expression, str) and not isinstance(expression, list):
            value = env.lookup(expression)
            if value is not None:
                return value
            else:
                # If not found as a variable, treat as string literal
                return UBPValue(expression, UBPType.STRING)
        
        # List evaluation (function calls)
        elif isinstance(expression, list) and len(expression) > 0:
            operator = expression[0]
            args = expression[1:]
            
            # Special forms (handled directly in evaluate function)
            if operator == 'quote':
                if len(args) != 1:
                    raise SyntaxError("quote requires exactly one argument")
                return UBPValue(args[0], UBPType.SYMBOL) # Quote returns the expression itself, not evaluated.
            
            elif operator in ['if', 'define', 'defun', 'lambda', 'let', 'cond']: # Special forms handled directly
                # For special forms, func_def['body'] will be a direct reference to a method like self._evaluate_if
                func_def = env.lookup_function(operator)
                if func_def and callable(func_def['body']):
                    return func_def['body'](args, env) # Call the method directly
                else:
                    raise NameError(f"Undefined special form: {operator}")
            
            # Function calls
            else:
                return self._evaluate_function_call(operator, args, env)
        
        else:
            raise SyntaxError(f"Invalid expression: {expression}")
    
    def _evaluate_function_call(self, operator: str, args: List[Any], 
                              env: UBPLispEnvironment) -> UBPValue:
        """Evaluate a function call"""
        # Look up function
        func_def = env.lookup_function(operator)
        if func_def is None:
            raise NameError(f"Undefined function: {operator}")
        
        # For built-in functions, func_def['body'] is a bound method (e.g., self._builtin_add).
        # We need to ensure we call it correctly.
        if callable(func_def['body']):
            # Dynamically get the method from 'self' to ensure proper binding if 'self' reference changed
            # This is a defensive measure for the 'attribute not found' error.
            method_name = func_def['body'].__name__ # Get the original method name (e.g., '_builtin_add')
            
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                # If it's a bound method, it should be callable.
                return func_def['body'](args, env) # Directly call the stored bound method
            else:
                raise AttributeError(f"UBPLispInterpreter object has no callable attribute '{method_name}' "
                                     f"or it's not bound correctly. Type of stored body: {type(func_def['body'])}")
        
        # Handle user-defined functions
        params = func_def['params']
        body = func_def['body']
        closure = func_def['closure']
        
        # Create new environment for function execution
        func_env = UBPLispEnvironment(closure)
        
        # Bind parameters
        if len(params) > 0 and params[0] == '&rest':
            # Variable arguments
            rest_param = params[1]
            evaluated_args = [self.evaluate(arg, env) for arg in args]
            func_env.define(rest_param, UBPValue(evaluated_args, UBPType.LIST))
        else:
            # Fixed arguments
            if len(args) != len(params):
                raise TypeError(f"Function {operator} expects {len(params)} arguments, got {len(args)}")
            
            for param, arg in zip(params, args):
                evaluated_arg = self.evaluate(arg, env)
                func_env.define(param, evaluated_arg)
        
        # Execute function body
        return self.evaluate(body, func_env)
    
    # Built-in function implementations
    def _builtin_add(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Addition function"""
        result = 0
        for arg in args:
            val = self.evaluate(arg, env)
            if val.ubp_type != UBPType.NUMBER:
                raise TypeError("Addition requires numeric arguments")
            result += val.value
        return UBPValue(result, UBPType.NUMBER)
    
    def _builtin_subtract(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Subtraction function"""
        if len(args) == 0:
            raise TypeError("Subtraction requires at least one argument")
        
        first_val = self.evaluate(args[0], env)
        if first_val.ubp_type != UBPType.NUMBER:
            raise TypeError("Subtraction requires numeric arguments")
        
        if len(args) == 1:
            return UBPValue(-first_val.value, UBPType.NUMBER)
        
        result = first_val.value
        for arg in args[1:]:
            val = self.evaluate(arg, env)
            if val.ubp_type != UBPType.NUMBER:
                raise TypeError("Subtraction requires numeric arguments")
            result -= val.value
        
        return UBPValue(result, UBPType.NUMBER)
    
    def _builtin_multiply(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Multiplication function"""
        result = 1
        for arg in args:
            val = self.evaluate(arg, env)
            if val.ubp_type != UBPType.NUMBER:
                raise TypeError("Multiplication requires numeric arguments")
            result *= val.value
        return UBPValue(result, UBPType.NUMBER)
    
    def _builtin_divide(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Division function"""
        if len(args) == 0:
            raise TypeError("Division requires at least one argument")
        
        first_val = self.evaluate(args[0], env)
        if first_val.ubp_type != UBPType.NUMBER:
            raise TypeError("Division requires numeric arguments")
        
        if len(args) == 1:
            if first_val.value == 0:
                raise ZeroDivisionError("Division by zero")
            return UBPValue(1.0 / first_val.value, UBPType.NUMBER)
        
        result = first_val.value
        for arg in args[1:]:
            val = self.evaluate(arg, env)
            if val.ubp_type != UBPType.NUMBER:
                raise TypeError("Division requires numeric arguments")
            if val.value == 0:
                raise ZeroDivisionError("Division by zero")
            result /= val.value
        
        return UBPValue(result, UBPType.NUMBER)

    def _builtin_equal(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Equality comparison"""
        if len(args) != 2:
            raise TypeError("= requires exactly two arguments")
        val1 = self.evaluate(args[0], env)
        val2 = self.evaluate(args[1], env)
        return UBPValue(val1.value == val2.value, UBPType.BOOLEAN)

    def _builtin_less(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Less than comparison"""
        if len(args) != 2:
            raise TypeError("< requires exactly two arguments")
        val1 = self.evaluate(args[0], env)
        val2 = self.evaluate(args[1], env)
        if val1.ubp_type != UBPType.NUMBER or val2.ubp_type != UBPType.NUMBER:
            raise TypeError("Comparison requires numeric arguments")
        return UBPValue(val1.value < val2.value, UBPType.BOOLEAN)

    def _builtin_greater(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Greater than comparison"""
        if len(args) != 2:
            raise TypeError("> requires exactly two arguments")
        val1 = self.evaluate(args[0], env)
        val2 = self.evaluate(args[1], env)
        if val1.ubp_type != UBPType.NUMBER or val2.ubp_type != UBPType.NUMBER:
            raise TypeError("Comparison requires numeric arguments")
        return UBPValue(val1.value > val2.value, UBPType.BOOLEAN)

    def _builtin_less_equal(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Less than or equal comparison"""
        if len(args) != 2:
            raise TypeError("<= requires exactly two arguments")
        val1 = self.evaluate(args[0], env)
        val2 = self.evaluate(args[1], env)
        if val1.ubp_type != UBPType.NUMBER or val2.ubp_type != UBPType.NUMBER:
            raise TypeError("Comparison requires numeric arguments")
        return UBPValue(val1.value <= val2.value, UBPType.BOOLEAN)

    def _builtin_greater_equal(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Greater than or equal comparison"""
        if len(args) != 2:
            raise TypeError(">= requires exactly two arguments")
        val1 = self.evaluate(args[0], env)
        val2 = self.evaluate(args[1], env)
        if val1.ubp_type != UBPType.NUMBER or val2.ubp_type != UBPType.NUMBER:
            raise TypeError("Comparison requires numeric arguments")
        return UBPValue(val1.value >= val2.value, UBPType.BOOLEAN)

    def _builtin_cons(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Constructs a list (cons cell)"""
        if len(args) != 2:
            raise TypeError("cons requires exactly two arguments")
        car_val = self.evaluate(args[0], env)
        cdr_val = self.evaluate(args[1], env)
        return UBPValue([car_val, cdr_val], UBPType.LIST)

    def _builtin_car(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Returns the first element of a list"""
        if len(args) != 1:
            raise TypeError("car requires exactly one argument")
        lst_val = self.evaluate(args[0], env)
        if lst_val.ubp_type != UBPType.LIST or not isinstance(lst_val.value, list) or not lst_val.value:
            raise TypeError("car requires a non-empty list")
        return lst_val.value[0] # Return the UBPValue of the first element

    def _builtin_cdr(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Returns the rest of the list"""
        if len(args) != 1:
            raise TypeError("cdr requires exactly one argument")
        lst_val = self.evaluate(args[0], env)
        if lst_val.ubp_type != UBPType.LIST or not isinstance(lst_val.value, list) or not lst_val.value:
            raise TypeError("cdr requires a non-empty list")
        return UBPValue(lst_val.value[1:], UBPType.LIST)

    def _builtin_list(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Creates a list from arguments"""
        evaluated_args = [self.evaluate(arg, env) for arg in args]
        return UBPValue(evaluated_args, UBPType.LIST)

    def _builtin_length(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Returns the length of a list"""
        if len(args) != 1:
            raise TypeError("length requires exactly one argument")
        lst_val = self.evaluate(args[0], env)
        if lst_val.ubp_type != UBPType.LIST or not isinstance(lst_val.value, list):
            raise TypeError("length requires a list")
        return UBPValue(len(lst_val.value), UBPType.NUMBER)
    
    def _builtin_make_offbit(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Create an OffBit"""
        if len(args) != 1:
            raise TypeError("make-offbit requires exactly one argument")
        
        val = self.evaluate(args[0], env)
        if val.ubp_type != UBPType.NUMBER:
            raise TypeError("OffBit value must be numeric")
        
        # Create 24-bit OffBit representation
        bit_value = int(val.value) & 0xFFFFFF  # Mask to 24 bits
        
        return UBPValue(bit_value, UBPType.OFFBIT, {'bits': 24})
    
    def _builtin_toggle(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Toggle an OffBit"""
        if len(args) != 1:
            raise TypeError("toggle requires exactly one argument")
        
        val = self.evaluate(args[0], env)
        if val.ubp_type != UBPType.OFFBIT:
            raise TypeError("toggle requires an OffBit")
        
        # XOR toggle operation
        toggled_value = val.value ^ 0xFFFFFF  # Toggle all 24 bits
        
        return UBPValue(toggled_value, UBPType.OFFBIT, val.metadata)
    
    def _builtin_resonance(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Compute resonance between frequencies"""
        if len(args) != 2:
            raise TypeError("resonance requires exactly two arguments")
        
        freq1_val = self.evaluate(args[0], env)
        freq2_val = self.evaluate(args[1], env)
        
        if freq1_val.ubp_type != UBPType.NUMBER or freq2_val.ubp_type != UBPType.NUMBER:
            raise TypeError("resonance requires numeric frequencies")
        
        freq1, freq2 = freq1_val.value, freq2_val.value
        
        # Compute resonance using UBP formula
        if freq1 == 0 or freq2 == 0:
            resonance_value = 0.0
        else:
            # Resonance strength based on frequency ratio
            ratio = min(freq1, freq2) / max(freq1, freq2)
            resonance_value = ratio * math.exp(-0.0002 * abs(freq1 - freq2)**2)
        
        return UBPValue(resonance_value, UBPType.FREQUENCY, 
                       {'freq1': freq1, 'freq2': freq2})

    def _builtin_entangle(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Entangle two OffBits (simplified)"""
        if len(args) != 2:
            raise TypeError("entangle requires exactly two OffBits")
        bit1_val = self.evaluate(args[0], env)
        bit2_val = self.evaluate(args[1], env)

        if bit1_val.ubp_type != UBPType.OFFBIT or bit2_val.ubp_type != UBPType.OFFBIT:
            raise TypeError("entangle requires OffBit arguments")
        
        # Simple entanglement: XOR and amplify by a coherence factor
        coherence_factor = _config.performance.COHERENCE_THRESHOLD # Use config threshold
        result_value = int(abs(bit1_val.value - bit2_val.value) * coherence_factor)
        result_value = max(0, min(result_value, 0xFFFFFF))
        
        return UBPValue(result_value, UBPType.OFFBIT, {'entangled': True, 'coherence_factor': coherence_factor})

    def _builtin_coherence(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Compute coherence of a bitfield or list of values"""
        if len(args) != 1:
            raise TypeError("coherence requires exactly one argument")
        
        data_val = self.evaluate(args[0], env)
        
        if data_val.ubp_type == UBPType.LIST and isinstance(data_val.value, list):
            values = np.array([v.value if isinstance(v, UBPValue) else v for v in data_val.value if isinstance(v, (int, float, UBPValue))])
        elif data_val.ubp_type == UBPType.OFFBIT: # If single OffBit, coherence is always 1
            values = np.array([data_val.value])
        elif data_val.ubp_type == UBPType.NUMBER:
             values = np.array([data_val.value])
        else:
            raise TypeError("coherence requires a LIST, OFFBIT or NUMBER argument")

        if len(values) < 2 or np.all(values == values[0]): # All same value or single value
            coherence_score = 1.0
        else:
            # Simplified statistical coherence calculation
            std_dev = np.std(values)
            mean_val = np.mean(values)
            if mean_val == 0:
                coherence_score = 1.0 if std_dev == 0 else 0.0
            else:
                coherence_score = 1.0 / (1.0 + (std_dev / abs(mean_val)))
        
        return UBPValue(coherence_score, UBPType.COHERENCE, {'source': data_val.ubp_type.value})
    
    def _builtin_spin_transition(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Compute spin transition using UBP formula"""
        if len(args) != 2:
            raise TypeError("spin-transition requires exactly two arguments")
        
        bit_val = self.evaluate(args[0], env)
        realm_val = self.evaluate(args[1], env)
        
        if bit_val.ubp_type not in [UBPType.NUMBER, UBPType.OFFBIT]:
            raise TypeError("spin-transition requires numeric bit value")
        
        if realm_val.ubp_type != UBPType.STRING:
            raise TypeError("spin-transition requires realm string")
        
        # Get bit value
        if bit_val.ubp_type == UBPType.OFFBIT:
            bit_value = bit_val.value / 0xFFFFFF  # Normalize to 0-1
        else:
            bit_value = bit_val.value
        
        # Get toggle probability for realm from config
        realm = realm_val.value.lower()
        p_s = _config.constants.UBP_TOGGLE_PROBABILITIES.get(realm, _config.constants.E / 12)
        
        # Compute spin transition: b_i × ln(1/p_s)
        transition_value = bit_value * math.log(1.0 / p_s)
        
        return UBPValue(transition_value, UBPType.NUMBER, 
                       {'realm': realm, 'toggle_probability': p_s})
    
    def _builtin_store(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Store value in BitBase"""
        if len(args) != 1:
            raise TypeError("store requires exactly one argument")
        
        val = self.evaluate(args[0], env)
        content_hash = self.bitbase.store(val.value, val.ubp_type, val.metadata)
        
        return UBPValue(content_hash, UBPType.STRING, {'stored_type': val.ubp_type.value})
    
    def _builtin_retrieve(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Retrieve value from BitBase"""
        if len(args) != 1:
            raise TypeError("retrieve requires exactly one argument")
        
        hash_val = self.evaluate(args[0], env)
        if hash_val.ubp_type != UBPType.STRING:
            raise TypeError("retrieve requires string hash")
        
        entry = self.bitbase.retrieve(hash_val.value)
        if entry is None:
            return UBPValue(None, UBPType.NIL)
        
        # When retrieving, wrap the content in UBPValue with its original type
        return UBPValue(entry.content, entry.ubp_type, entry.metadata)
    
    def _builtin_cond(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluates a series of clauses until one is true."""
        for clause in args:
            if not isinstance(clause, list) or len(clause) < 1:
                raise SyntaxError("cond clauses must be lists with at least a condition")
            
            condition_expr = clause[0]
            if condition_expr == 'else': # 'else' clause
                if len(clause) < 2:
                    raise SyntaxError("else clause must have an expression")
                return self.evaluate(clause[1], env)
            
            condition_val = self.evaluate(condition_expr, env)
            is_true = (condition_val.ubp_type != UBPType.NIL and 
                       condition_val.ubp_type != UBPType.BOOLEAN or 
                       condition_val.value is not False)
            
            if is_true:
                if len(clause) == 1: # Condition is true, but no body (e.g. (cond (#t)))
                    return UBPValue(condition_val.value, condition_val.ubp_type)
                return self.evaluate(clause[1], env) # Execute first body expression
        
        return UBPValue(None, UBPType.NIL) # No condition met

    def _evaluate_if(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluate if expression"""
        if len(args) < 2 or len(args) > 3:
            raise SyntaxError("if requires 2 or 3 arguments")
        
        condition = self.evaluate(args[0], env)
        
        # Check truthiness
        is_true = (condition.ubp_type != UBPType.NIL and 
                  condition.ubp_type != UBPType.BOOLEAN or 
                  condition.value is not False)
        
        if is_true:
            return self.evaluate(args[1], env)
        elif len(args) == 3:
            return self.evaluate(args[2], env)
        else:
            return UBPValue(None, UBPType.NIL)
    
    def _evaluate_define(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluate define expression"""
        if len(args) != 2:
            raise SyntaxError("define requires exactly 2 arguments")
        
        symbol = args[0]
        if not isinstance(symbol, str):
            raise SyntaxError("define requires symbol as first argument")
        
        value = self.evaluate(args[1], env)
        env.define(symbol, value)
        
        return UBPValue(symbol, UBPType.SYMBOL)
    
    def _evaluate_defun(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluate defun expression"""
        if len(args) != 3:
            raise SyntaxError("defun requires exactly 3 arguments")
        
        name = args[0]
        params = args[1]
        body = args[2]
        
        if not isinstance(name, str):
            raise SyntaxError("defun requires symbol as function name")
        
        if not isinstance(params, list):
            raise SyntaxError("defun requires list as parameters")
        
        env.define_function(name, params, body)
        
        return UBPValue(name, UBPType.FUNCTION)
    
    def _evaluate_lambda(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluate lambda expression"""
        if len(args) != 2:
            raise SyntaxError("lambda requires exactly 2 arguments")
        
        params = args[0]
        body = args[1]
        
        if not isinstance(params, list):
            raise SyntaxError("lambda requires list as parameters")
        
        # Create lambda function
        lambda_func = {
            'params': params,
            'body': body,
            'closure': env,
            'type': 'lambda'
        }
        
        return UBPValue(lambda_func, UBPType.FUNCTION)
    
    def _evaluate_let(self, args: List[Any], env: UBPLispEnvironment) -> UBPValue:
        """Evaluate let expression"""
        if len(args) != 2:
            raise SyntaxError("let requires exactly 2 arguments")
        
        bindings = args[0]
        body = args[1]
        
        if not isinstance(bindings, list):
            raise SyntaxError("let requires list as bindings")
        
        # Create new environment
        let_env = UBPLispEnvironment(env)
        
        # Process bindings
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2:
                raise SyntaxError("let binding must be [symbol value]")
            
            symbol, value_expr = binding
            if not isinstance(symbol, str):
                raise SyntaxError("let binding symbol must be string")
            
            value = self.evaluate(value_expr, env)
            let_env.define(symbol, value)
        
        # Evaluate body in new environment
        return self.evaluate(body, let_env)
    
    def run(self, source_code: str) -> UBPValue:
        """
        Run UBP-Lisp source code.
        
        Args:
            source_code: UBP-Lisp source code
        
        Returns:
            Result of evaluation
        """
        try:
            # Tokenize
            tokens = self.parser.tokenize(source_code)
            
            # Parse
            ast = self.parser.parse(tokens)
            
            # Evaluate
            result = self.evaluate(ast)
            
            return result
            
        except Exception as e:
            return UBPValue(f"Error: {str(e)}", UBPType.STRING, {'error': True})
    
    def validate_ubp_lisp_system(self) -> Dict[str, Any]:
        """
        Validate the UBP-Lisp system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'parser_functionality': True,
            'basic_evaluation': True,
            'ubp_operations': True,
            'bitbase_integration': True,
            'function_definition': True
        }
        
        try:
            # Test 1: Parser functionality
            test_code = "(+ 1 2 3)"
            tokens = self.parser.tokenize(test_code)
            ast = self.parser.parse(tokens)
            
            if not isinstance(ast, list) or ast[0] != '+':
                validation_results['parser_functionality'] = False
                validation_results['parser_error'] = "Parser failed to parse basic expression"
            
            # Test 2: Basic evaluation
            result = self.run("(+ 1 2 3)")
            
            if result.ubp_type != UBPType.NUMBER or result.value != 6:
                validation_results['basic_evaluation'] = False
                validation_results['evaluation_error'] = f"Expected 6, got {result.value}"
            
            # Test 3: UBP operations
            offbit_result = self.run("(make-offbit 42)")
            
            if offbit_result.ubp_type != UBPType.OFFBIT:
                validation_results['ubp_operations'] = False
                validation_results['ubp_error'] = "OffBit creation failed"
            
            # Test 4: BitBase integration
            # Clear HexDictionary for a clean test
            self.bitbase.hex_dict.clear_all() 
            store_result = self.run("(store 123)")
            retrieve_result = self.run(f"(retrieve \"{store_result.value}\")")
            
            if retrieve_result.ubp_type != UBPType.NUMBER or retrieve_result.value != 123:
                validation_results['bitbase_integration'] = False
                validation_results['bitbase_error'] = "BitBase store/retrieve failed"
            
            # Test 5: Function definition
            self.run("(defun square (x) (* x x))")
            square_result = self.run("(square 5)")
            
            if square_result.ubp_type != UBPType.NUMBER or square_result.value != 25:
                validation_results['function_definition'] = False
                validation_results['function_error'] = "Function definition failed"
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['parser_functionality'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_ubp_lisp_interpreter(hex_dict_instance: Optional[HexDictionary] = None) -> UBPLispInterpreter:
    """
    Create a UBP-Lisp interpreter with default configuration.
    
    Returns:
        Configured UBPLispInterpreter instance
    """
    return UBPLispInterpreter(hex_dict_instance=hex_dict_instance)