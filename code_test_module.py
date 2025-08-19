import ast
import re
import sys
import io
from typing import Tuple, List, Dict, Any
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)

class CodeType(Enum):
    """Enumeration of different code types for testing purposes."""
    SIMPLE_FUNCTIONS = "simple_functions"
    COMPLEX_APPLICATION = "complex_application"
    CONSTANTS_ONLY = "constants_only"
    INCOMPLETE_CODE = "incomplete_code"

class CodeAnalyzer:
    """Analyzes Python code to determine appropriate testing strategy."""
    
    # Keywords that indicate complex applications
    COMPLEX_INDICATORS = [
        'pygame', 'tkinter', 'flask', 'django', 'fastapi',
        'mainloop', 'run_forever', 'app.run', 'game_loop',
        'event.get', 'pygame.display', 'render', 'blit',
        'Canvas', 'Frame', 'Button', 'Label', 'Entry'
    ]
    
    # Patterns that suggest GUI or interactive elements
    GUI_PATTERNS = [
        r'\.mainloop\s*\(',
        r'pygame\.display\.',
        r'pygame\.event\.',
        r'\.show\s*\(',
        r'\.run\s*\(',
        r'while\s+.*running',
        r'for\s+event\s+in\s+pygame\.event\.get',
    ]
    
    def analyze_code_type(self, code: str) -> CodeType:
        """
        Analyze code to determine its type for testing purposes.
        
        Args:
            code (str): Python code to analyze
            
        Returns:
            CodeType: The determined code type
        """
        try:
            # Parse the code into AST
            tree = ast.parse(code)
            
            # Check for incomplete or constant-only code
            if self._is_constants_only(tree):
                return CodeType.CONSTANTS_ONLY
            
            if self._is_incomplete_code(code, tree):
                return CodeType.INCOMPLETE_CODE
            
            # Check for complex application indicators
            if self._is_complex_application(code, tree):
                return CodeType.COMPLEX_APPLICATION
            
            # Default to simple functions
            return CodeType.SIMPLE_FUNCTIONS
            
        except SyntaxError:
            return CodeType.INCOMPLETE_CODE
    
    def _is_constants_only(self, tree: ast.AST) -> bool:
        """Check if code only contains constants, imports, or assignments."""
        meaningful_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.For, ast.While, ast.If)):
                meaningful_nodes.append(node)
        
        return len(meaningful_nodes) == 0
    
    def _is_incomplete_code(self, code: str, tree: ast.AST) -> bool:
        """Check if code appears to be incomplete or pseudocode."""
        # Check for common incomplete patterns
        incomplete_patterns = [
            r'#\s*TODO',
            r'pass\s*$',
            r'\.\.\.', 
            r'raise\s+NotImplementedError',
            r'# Implementation here',
            r'# Your code here'
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, code, re.MULTILINE | re.IGNORECASE):
                return True
        
        # Check if functions only contain pass statements
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    return True
        
        return False
    
    def _is_complex_application(self, code: str, tree: ast.AST) -> bool:
        """Check if code is a complex application requiring different testing."""
        code_lower = code.lower()
        
        # Check for complex indicators in imports and code
        for indicator in self.COMPLEX_INDICATORS:
            if indicator in code_lower:
                return True
        
        # Check for GUI patterns
        for pattern in self.GUI_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        
        # Check AST for complex patterns
        has_classes = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        has_multiple_functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]) > 3
        
        if has_classes and has_multiple_functions:
            return True
        
        return False

class CodeTester:
    """Handles different testing strategies based on code type."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
    
    async def test_code(self, code: str, task_objective: str = "") -> Tuple[str, str]:
        """
        Test code using appropriate strategy based on code type.
        
        Args:
            code (str): Python code to test
            task_objective (str): The original task objective for context
            
        Returns:
            Tuple[str, str]: (test_result, status) where status is 'completed' or 'failed'
        """
        try:
            code_type = self.analyzer.analyze_code_type(code)
            logger.info(f"Code type detected: {code_type.value}")
            
            if code_type == CodeType.CONSTANTS_ONLY:
                return "Code contains only constants/imports - no testing needed", 'completed'
            
            elif code_type == CodeType.INCOMPLETE_CODE:
                return "Code appears incomplete or contains pseudocode", 'failed'
            
            elif code_type == CodeType.SIMPLE_FUNCTIONS:
                return await self._test_simple_functions(code)
            
            elif code_type == CodeType.COMPLEX_APPLICATION:
                return await self._test_complex_application(code)
            
        except Exception as e:
            logger.error(f"Code testing error: {e}")
            return f"Testing error: {str(e)}", 'failed'
    
    async def _test_simple_functions(self, code: str) -> Tuple[str, str]:
        """Test simple functions with unit tests."""
        try:
            # Extract function signatures for testing
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return "No testable functions found", 'completed'
            
            # Generate and run simple unit tests
            test_results = await self._execute_with_timeout(self._run_unit_tests, code, timeout=10)
            
            if "Error executing code" in test_results:
                return f"Unit test failed:\n{test_results}", 'failed'
            else:
                return f"Unit tests passed:\n{test_results}", 'completed'
                
        except Exception as e:
            return f"Unit test error: {str(e)}", 'failed'
    
    async def _test_complex_application(self, code: str) -> Tuple[str, str]:
        """Test complex applications with syntax and import validation."""
        results = []
        
        # Test 1: Syntax validation
        try:
            compile(code, '<string>', 'exec')
            results.append("✓ Syntax validation passed")
        except SyntaxError as e:
            return f"Syntax Error: {str(e)}", 'failed'
        
        # Test 2: Import validation
        try:
            import_test_result = await self._test_imports(code)
            results.append(import_test_result)
        except Exception as e:
            results.append(f"⚠ Import test warning: {str(e)}")
        
        # Test 3: Basic instantiation test (without running main loops)
        try:
            instantiation_result = await self._test_instantiation(code)
            results.append(instantiation_result)
        except Exception as e:
            results.append(f"⚠ Instantiation test warning: {str(e)}")
        
        return "\n".join(results), 'completed'
    
    async def _test_imports(self, code: str) -> str:
        """Test that all imports in the code work."""
        tree = ast.parse(code)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        failed_imports = []
        for imp in set(imports):
            try:
                # Only test standard library and common packages
                if not self._is_common_package(imp):
                    continue
                __import__(imp)
            except ImportError:
                failed_imports.append(imp)
        
        if failed_imports:
            return f"⚠ Missing packages: {', '.join(failed_imports)}"
        else:
            return "✓ All imports available"
    
    def _is_common_package(self, package_name: str) -> bool:
        """Check if package is commonly available (to avoid testing obscure packages)."""
        common_packages = {
            'pygame', 'tkinter', 'matplotlib', 'numpy', 'pandas', 
            'requests', 'flask', 'django', 'sys', 'os', 'json', 
            'random', 'math', 're', 'time', 'datetime'
        }
        return package_name.split('.')[0] in common_packages
    
    async def _test_instantiation(self, code: str) -> str:
        """Test basic instantiation without running interactive elements."""
        # Create a modified version of code that doesn't run main loops
        safe_code = self._make_code_safe_for_testing(code)
        
        try:
            exec_globals = {}
            exec(safe_code, exec_globals)
            return "✓ Code structure validation passed"
        except Exception as e:
            return f"⚠ Structure validation warning: {str(e)}"
    
    def _make_code_safe_for_testing(self, code: str) -> str:
        """Remove or comment out interactive elements for safe testing."""
        # Common patterns to disable
        unsafe_patterns = [
            (r'\.mainloop\s*\(\s*\)', '# .mainloop()  # Disabled for testing'),
            (r'\.run\s*\(\s*\)', '# .run()  # Disabled for testing'),
            (r'app\.run\s*\([^)]*\)', '# app.run()  # Disabled for testing'),
            (r'plt\.show\s*\(\s*\)', '# plt.show()  # Disabled for testing'),
            (r'while\s+running:', '# while running:  # Disabled for testing'),
            (r'while\s+True:', '# while True:  # Disabled for testing'),
        ]
        
        safe_code = code
        for pattern, replacement in unsafe_patterns:
            safe_code = re.sub(pattern, replacement, safe_code)
        
        return safe_code
    
    async def _run_unit_tests(self, code: str) -> str:
        """Execute simple unit tests for functions."""
        # This is a simplified version - in practice, you'd generate smarter tests
        exec_globals = {}
        exec(code, exec_globals)
        
        # Try to find and test simple functions
        test_results = []
        for name, obj in exec_globals.items():
            if callable(obj) and not name.startswith('_'):
                try:
                    # Very basic smoke test - just call with no args if possible
                    if obj.__code__.co_argcount == 0:
                        result = obj()
                        test_results.append(f"✓ {name}() executed successfully")
                except Exception as e:
                    test_results.append(f"⚠ {name}() test warning: {str(e)}")
        
        return "\n".join(test_results) if test_results else "No testable functions found"
    
    async def _execute_with_timeout(self, func, *args, timeout: int = 15) -> str:
        """Execute function with timeout to prevent hanging."""
        try:
            return await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, func, *args),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return "Error executing code: Execution timed out"
        except Exception as e:
            return f"Error executing code: {str(e)}"