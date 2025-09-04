"""
Universal Binary Principle (UBP) Framework v3.2+ - Hardware Profiles
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Hardware Profiles provides optimized configurations for different deployment
environments including 8GB iMac, 4GB mobile devices, Raspberry Pi 5, Kaggle,
Google Colab, and high-performance computing systems.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import platform
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("Warning: psutil not available. Hardware detection will be limited.")
    PSUTIL_AVAILABLE = False

# Import system constants
from system_constants import UBPConstants

@dataclass
class HardwareProfile:
    """Hardware profile configuration for UBP Framework deployment."""
    
    name: str
    description: str
    
    # Memory configuration
    total_memory_gb: float
    available_memory_gb: float
    
    # Processing configuration
    cpu_cores: int
    cpu_frequency_ghz: float
    
    # UBP-specific configuration
    max_offbits: int
    bitfield_dimensions: Tuple[int, ...]
    sparsity_level: float
    target_operations_per_second: int
    
    # Optional configuration with defaults
    memory_safety_factor: float = 0.8
    has_gpu: bool = False
    gpu_memory_gb: float = 0.0
    max_operation_time_seconds: float = 30.0
    
    # Error correction settings
    enable_error_correction: bool = True
    error_correction_level: str = "standard"  # "basic", "standard", "advanced"
    enable_padic_encoding: bool = True
    enable_fibonacci_encoding: bool = True
    
    # Optimization settings
    enable_parallel_processing: bool = True
    enable_gpu_acceleration: bool = False
    enable_memory_optimization: bool = True
    enable_sparse_matrices: bool = True
    
    # Environment-specific settings
    environment_type: str = "local"  # "local", "colab", "kaggle", "cloud"
    data_directory: str = "./data"
    output_directory: str = "./output"
    temp_directory: str = "./temp"
    
    # Validation settings
    validation_iterations: int = 1000
    enable_extensive_testing: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class HardwareProfileManager:
    """
    Hardware Profile Manager for UBP Framework v3.0.
    
    Manages hardware-specific configurations and automatically detects
    optimal settings for different deployment environments.
    """
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        self.current_profile = None
        self.auto_detected_profile = None
    
    def _initialize_profiles(self) -> Dict[str, HardwareProfile]:
        """Initialize all predefined hardware profiles."""
        profiles = {}
        
        # 8GB iMac Profile
        profiles["8gb_imac"] = HardwareProfile(
            name="8GB iMac",
            description="Apple iMac with 8GB RAM - High performance configuration",
            total_memory_gb=8.0,
            available_memory_gb=6.0,
            memory_safety_factor=0.75,
            cpu_cores=8,
            cpu_frequency_ghz=3.2,
            has_gpu=True,
            gpu_memory_gb=2.0,
            max_offbits=UBPConstants.OFFBITS_8GB_IMAC,
            bitfield_dimensions=UBPConstants.BITFIELD_6D_FULL,
            sparsity_level=0.01,
            target_operations_per_second=8000,
            max_operation_time_seconds=0.5,
            error_correction_level="advanced",
            enable_gpu_acceleration=True,
            enable_extensive_testing=True,
            validation_iterations=10000,
            metadata={
                "platform": "darwin",
                "architecture": "x86_64",
                "optimization_level": "maximum"
            }
        )
        
        # Raspberry Pi 5 Profile
        profiles["raspberry_pi5"] = HardwareProfile(
            name="Raspberry Pi 5",
            description="Raspberry Pi 5 with 8GB RAM - Balanced performance",
            total_memory_gb=8.0,
            available_memory_gb=6.0,
            memory_safety_factor=0.8,
            cpu_cores=4,
            cpu_frequency_ghz=2.4,
            has_gpu=False,
            gpu_memory_gb=0.0,
            max_offbits=UBPConstants.OFFBITS_RASPBERRY_PI5,
            bitfield_dimensions=UBPConstants.BITFIELD_6D_MEDIUM,
            sparsity_level=0.01,
            target_operations_per_second=5000,
            max_operation_time_seconds=2.0,
            error_correction_level="standard",
            enable_gpu_acceleration=False,
            enable_memory_optimization=True,
            validation_iterations=5000,
            metadata={
                "platform": "linux",
                "architecture": "aarch64",
                "optimization_level": "balanced"
            }
        )
        
        # 4GB Mobile Profile
        profiles["4gb_mobile"] = HardwareProfile(
            name="4GB Mobile Device",
            description="Mobile device with 4GB RAM - Memory optimized",
            total_memory_gb=4.0,
            available_memory_gb=2.5,
            memory_safety_factor=0.9,
            cpu_cores=4,
            cpu_frequency_ghz=2.0,
            has_gpu=False,
            gpu_memory_gb=0.0,
            max_offbits=UBPConstants.OFFBITS_4GB_MOBILE,
            bitfield_dimensions=UBPConstants.BITFIELD_6D_SMALL,
            sparsity_level=0.001,
            target_operations_per_second=2000,
            max_operation_time_seconds=5.0,
            error_correction_level="basic",
            enable_parallel_processing=False,
            enable_memory_optimization=True,
            enable_sparse_matrices=True,
            validation_iterations=1000,
            metadata={
                "platform": "android",
                "architecture": "arm64",
                "optimization_level": "memory"
            }
        )
        
        # Google Colab Profile
        profiles["google_colab"] = HardwareProfile(
            name="Google Colab",
            description="Google Colab environment - GPU accelerated",
            total_memory_gb=12.0,
            available_memory_gb=10.0,
            memory_safety_factor=0.8,
            cpu_cores=2,
            cpu_frequency_ghz=2.3,
            has_gpu=True,
            gpu_memory_gb=15.0,
            max_offbits=500000,  # Optimized for Colab
            bitfield_dimensions=(120, 120, 120, 5, 2, 2),
            sparsity_level=0.01,
            target_operations_per_second=10000,
            max_operation_time_seconds=1.0,
            error_correction_level="advanced",
            enable_gpu_acceleration=True,
            enable_parallel_processing=True,
            environment_type="colab",
            data_directory="/content/data",
            output_directory="/content/output",
            temp_directory="/tmp",
            validation_iterations=5000,
            metadata={
                "platform": "linux",
                "architecture": "x86_64",
                "optimization_level": "gpu_accelerated",
                "cloud_provider": "google"
            }
        )
        
        # Kaggle Profile
        profiles["kaggle"] = HardwareProfile(
            name="Kaggle",
            description="Kaggle competition environment - Competition optimized",
            total_memory_gb=16.0,
            available_memory_gb=13.0,
            memory_safety_factor=0.8,
            cpu_cores=4,
            cpu_frequency_ghz=2.0,
            has_gpu=True,
            gpu_memory_gb=16.0,
            max_offbits=300000,  # Optimized for Kaggle
            bitfield_dimensions=(100, 100, 100, 5, 2, 2),
            sparsity_level=0.01,
            target_operations_per_second=8000,
            max_operation_time_seconds=1.5,
            error_correction_level="standard",
            enable_gpu_acceleration=True,
            environment_type="kaggle",
            data_directory="/kaggle/input",
            output_directory="/kaggle/working",
            temp_directory="/tmp",
            validation_iterations=3000,
            metadata={
                "platform": "linux",
                "architecture": "x86_64",
                "optimization_level": "competition",
                "cloud_provider": "kaggle"
            }
        )
        
        # High-Performance Computing Profile
        profiles["hpc"] = HardwareProfile(
            name="High-Performance Computing",
            description="HPC cluster or workstation - Maximum performance",
            total_memory_gb=64.0,
            available_memory_gb=56.0,
            memory_safety_factor=0.7,
            cpu_cores=32,
            cpu_frequency_ghz=3.5,
            has_gpu=True,
            gpu_memory_gb=48.0,
            max_offbits=10000000,  # 10M OffBits
            bitfield_dimensions=(300, 300, 300, 5, 2, 2),
            sparsity_level=0.1,
            target_operations_per_second=50000,
            max_operation_time_seconds=0.1,
            error_correction_level="advanced",
            enable_gpu_acceleration=True,
            enable_parallel_processing=True,
            enable_extensive_testing=True,
            validation_iterations=50000,
            metadata={
                "platform": "linux",
                "architecture": "x86_64",
                "optimization_level": "maximum_performance",
                "cluster_capable": True
            }
        )
        
        # Development Profile (for testing)
        profiles["development"] = HardwareProfile(
            name="Development",
            description="Development and testing environment - Fast iteration",
            total_memory_gb=8.0,
            available_memory_gb=6.0,
            memory_safety_factor=0.9,
            cpu_cores=4,
            cpu_frequency_ghz=2.5,
            has_gpu=False,
            gpu_memory_gb=0.0,
            max_offbits=10000,  # Small for fast testing
            bitfield_dimensions=(20, 20, 20, 5, 2, 2),
            sparsity_level=0.1,
            target_operations_per_second=1000,
            max_operation_time_seconds=10.0,
            error_correction_level="basic",
            enable_parallel_processing=False,
            validation_iterations=100,
            metadata={
                "platform": "any",
                "architecture": "any",
                "optimization_level": "development",
                "fast_iteration": True
            }
        )
        
        return profiles
    
    def auto_detect_profile(self) -> str:
        """
        Automatically detect the best hardware profile for the current environment.
        
        Returns:
            Profile name that best matches the current hardware
        """
        # Get system information
        if PSUTIL_AVAILABLE:
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
        else:
            # Fallback values if psutil is not available
            total_memory_gb = 8.0 # Assume a reasonable default for typical environments
            cpu_count = os.cpu_count() if os.cpu_count() is not None else 4 # Get logical cores, or default to 4
            print(f"Using fallback system info: Memory={total_memory_gb}GB, CPU Cores={cpu_count}")

        platform_system = platform.system().lower()
        
        # Check for cloud environments
        if self._is_google_colab():
            self.auto_detected_profile = "google_colab"
            return "google_colab"
        
        if self._is_kaggle():
            self.auto_detected_profile = "kaggle"
            return "kaggle"
        
        # Check for specific hardware configurations
        if total_memory_gb >= 32 and cpu_count >= 16:
            self.auto_detected_profile = "hpc"
            return "hpc"
        
        if total_memory_gb >= 7 and cpu_count >= 6 and platform_system == "darwin":
            self.auto_detected_profile = "8gb_imac"
            return "8gb_imac"
        
        if total_memory_gb >= 6 and cpu_count >= 4 and platform_system == "linux":
            # Could be Raspberry Pi 5 or similar
            if self._is_raspberry_pi():
                self.auto_detected_profile = "raspberry_pi5"
                return "raspberry_pi5"
        
        if total_memory_gb <= 5:
            self.auto_detected_profile = "4gb_mobile"
            return "4gb_mobile"
        
        # Default fallback
        self.auto_detected_profile = "development"
        return "development"
    
    def get_profile(self, profile_name: Optional[str] = None) -> HardwareProfile:
        """
        Get hardware profile by name or auto-detect.
        
        Args:
            profile_name: Name of the profile to get, or None for auto-detection
            
        Returns:
            HardwareProfile object
        """
        if profile_name is None:
            profile_name = self.auto_detect_profile()
        
        if profile_name not in self.profiles:
            raise ValueError(f"Unknown profile: {profile_name}. "
                           f"Available profiles: {list(self.profiles.keys())}")
        
        profile = self.profiles[profile_name]
        self.current_profile = profile
        return profile
    
    def list_profiles(self) -> Dict[str, str]:
        """
        List all available profiles with descriptions.
        
        Returns:
            Dictionary mapping profile names to descriptions
        """
        return {name: profile.description for name, profile in self.profiles.items()}
    
    def validate_profile(self, profile: HardwareProfile) -> Dict[str, bool]:
        """
        Validate that a hardware profile is suitable for the current system.
        
        Args:
            profile: Hardware profile to validate
            
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Memory validation
        if PSUTIL_AVAILABLE:
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            validations['sufficient_memory'] = system_memory_gb >= profile.total_memory_gb * 0.8
        else:
            validations['sufficient_memory'] = True # Assume sufficient if cannot detect

        # CPU validation
        system_cpu_count = os.cpu_count() if os.cpu_count() is not None else 4
        validations['sufficient_cpu'] = system_cpu_count >= profile.cpu_cores * 0.5
        
        # OffBit count validation
        estimated_memory_usage = self._estimate_memory_usage(profile)
        # Use a safe estimate if psutil not available
        available_memory = (psutil.virtual_memory().total if PSUTIL_AVAILABLE else 8 * (1024**3)) * profile.memory_safety_factor
        validations['memory_within_limits'] = estimated_memory_usage <= available_memory
        
        # Performance validation
        validations['reasonable_targets'] = (
            profile.target_operations_per_second <= 100000 and
            profile.max_operation_time_seconds >= 0.01
        )
        
        return validations
    
    def optimize_profile_for_system(self, base_profile_name: str) -> HardwareProfile:
        """
        Optimize a profile for the current system capabilities.
        
        Args:
            base_profile_name: Name of the base profile to optimize
            
        Returns:
            Optimized HardwareProfile
        """
        base_profile = self.profiles[base_profile_name]
        
        # Get system capabilities
        if PSUTIL_AVAILABLE:
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
        else:
            system_memory_gb = 8.0 # Fallback
        
        system_cpu_count = os.cpu_count() if os.cpu_count() is not None else 4
        
        # Create optimized profile
        optimized_profile = HardwareProfile(
            name=f"{base_profile.name} (Optimized)",
            description=f"{base_profile.description} - System optimized",
            total_memory_gb=min(base_profile.total_memory_gb, system_memory_gb),
            available_memory_gb=min(base_profile.available_memory_gb, system_memory_gb * 0.8),
            memory_safety_factor=base_profile.memory_safety_factor,
            cpu_cores=min(base_profile.cpu_cores, system_cpu_count),
            cpu_frequency_ghz=base_profile.cpu_frequency_ghz,
            has_gpu=base_profile.has_gpu,
            gpu_memory_gb=base_profile.gpu_memory_gb,
            max_offbits=self._optimize_offbit_count(base_profile, system_memory_gb),
            bitfield_dimensions=self._optimize_bitfield_dimensions(base_profile, system_memory_gb),
            sparsity_level=base_profile.sparsity_level,
            target_operations_per_second=base_profile.target_operations_per_second,
            max_operation_time_seconds=base_profile.max_operation_time_seconds,
            error_correction_level=base_profile.error_correction_level,
            enable_padic_encoding=base_profile.enable_padic_encoding,
            enable_fibonacci_encoding=base_profile.enable_fibonacci_encoding,
            enable_parallel_processing=base_profile.enable_parallel_processing and system_cpu_count > 1,
            enable_gpu_acceleration=base_profile.enable_gpu_acceleration,
            enable_memory_optimization=True,  # Always enable for optimized profiles
            enable_sparse_matrices=True,
            environment_type=base_profile.environment_type,
            data_directory=base_profile.data_directory,
            output_directory=base_profile.output_directory,
            temp_directory=base_profile.temp_directory,
            validation_iterations=base_profile.validation_iterations,
            enable_extensive_testing=base_profile.enable_extensive_testing,
            metadata={
                **base_profile.metadata,
                "optimized_for_system": True,
                "system_memory_gb": system_memory_gb,
                "system_cpu_count": system_cpu_count
            }
        )
        
        return optimized_profile
    
    def get_environment_config(self, profile: HardwareProfile) -> Dict[str, Any]:
        """
        Get environment-specific configuration for a profile.
        
        Args:
            profile: Hardware profile
            
        Returns:
            Environment configuration dictionary
        """
        config = {
            "directories": {
                "data": profile.data_directory,
                "output": profile.output_directory,
                "temp": profile.temp_directory
            },
            "memory": {
                "total_gb": profile.total_memory_gb,
                "available_gb": profile.available_memory_gb,
                "safety_factor": profile.memory_safety_factor
            },
            "processing": {
                "cpu_cores": profile.cpu_cores,
                "enable_parallel": profile.enable_parallel_processing,
                "enable_gpu": profile.enable_gpu_acceleration,
                "gpu_memory_gb": profile.gpu_memory_gb
            },
            "ubp_settings": {
                "max_offbits": profile.max_offbits,
                "bitfield_dimensions": profile.bitfield_dimensions,
                "sparsity_level": profile.sparsity_level,
                "error_correction_level": profile.error_correction_level
            },
            "performance": {
                "target_ops_per_second": profile.target_operations_per_second,
                "max_operation_time": profile.max_operation_time_seconds,
                "validation_iterations": profile.validation_iterations
            },
            "optimization": {
                "enable_memory_optimization": profile.enable_memory_optimization,
                "enable_sparse_matrices": profile.enable_sparse_matrices,
                "enable_padic_encoding": profile.enable_padic_encoding,
                "enable_fibonacci_encoding": profile.enable_fibonacci_encoding
            }
        }
        
        return config
    
    def _is_google_colab(self) -> bool:
        """Check if running in Google Colab."""
        return 'COLAB_GPU' in os.environ # More robust check for Colab
    
    def _is_kaggle(self) -> bool:
        """Check if running in Kaggle environment."""
        return os.path.exists('/kaggle')
    
    def _is_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'raspberry pi' in cpuinfo.lower() or 'bcm2835' in cpuinfo.lower() # More general
        except:
            return False
    
    def _estimate_memory_usage(self, profile: HardwareProfile) -> float:
        """
        Estimate memory usage for a profile configuration.
        
        Args:
            profile: Hardware profile
            
        Returns:
            Estimated memory usage in bytes
        """
        # Estimate OffBit memory usage (32 bits per OffBit)
        offbit_memory = profile.max_offbits * 4  # 4 bytes per OffBit
        
        # Estimate Bitfield memory usage
        bitfield_cells = np.prod(profile.bitfield_dimensions)
        bitfield_memory = bitfield_cells * 4  # 4 bytes per cell
        
        # Estimate additional overhead (matrices, error correction, etc.)
        overhead_factor = 2.0 if profile.enable_sparse_matrices else 3.0
        
        total_memory = (offbit_memory + bitfield_memory) * overhead_factor
        
        return total_memory
    
    def _optimize_offbit_count(self, base_profile: HardwareProfile, system_memory_gb: float) -> int:
        """Optimize OffBit count for system memory."""
        available_memory_bytes = system_memory_gb * base_profile.memory_safety_factor * (1024**3)
        
        # Estimate memory per OffBit (including overhead)
        memory_per_offbit = 4 * 2.5  # 4 bytes + 150% overhead
        
        max_offbits_by_memory = int(available_memory_bytes * 0.5 / memory_per_offbit)
        
        return min(base_profile.max_offbits, max_offbits_by_memory)
    
    def _optimize_bitfield_dimensions(self, base_profile: HardwareProfile, 
                                    system_memory_gb: float) -> Tuple[int, ...]:
        """Optimize Bitfield dimensions for system memory."""
        base_dims = base_profile.bitfield_dimensions
        
        # If system has less memory, scale down dimensions proportionally
        memory_ratio = system_memory_gb / base_profile.total_memory_gb
        
        if memory_ratio < 0.8:
            # Scale down dimensions
            scale_factor = memory_ratio ** (1/3)  # Cube root for 3D scaling
            
            new_dims = tuple(
                max(10, int(dim * scale_factor)) if i < 3 else dim
                for i, dim in enumerate(base_dims)
            )
            
            return new_dims
        
        return base_dims

# Create global instance
HARDWARE_MANAGER = HardwareProfileManager()
