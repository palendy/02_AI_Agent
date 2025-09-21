#!/usr/bin/env python3
"""
Main run script for SRS Generation Agent.
Provides both programmatic and command-line interfaces for SRS generation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from config import get_config
from srs_generation_agent import SRSGenerationAgent
from validation_engine import ValidationEngine
from logger import setup_logging, get_logger, ComponentType, log_operation
from cli import cli


def main():
    """Main entry point for the application."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="SRS Generation Agent - Generate Software Requirements Specifications for semiconductor firmware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate SRS from documents
  python run.py --project "Motor Control" --input spec1.pdf spec2.docx --output motor_srs.md
  
  # Use CLI interface
  python run.py --cli
  
  # Interactive generation
  python run.py --interactive --project "Sensor Hub" --input requirements.txt
  
  # Validate existing SRS
  python run.py --validate existing_srs.md --level comprehensive
        """
    )
    
    # Core options
    parser.add_argument('--project', '-p', type=str, help='Project name')
    parser.add_argument('--input', '-i', nargs='+', help='Input document files')
    parser.add_argument('--output', '-o', type=str, help='Output SRS file path')
    
    # Configuration options
    parser.add_argument('--architecture', '-a', default='ARM Cortex-M', 
                       help='Target architecture (default: ARM Cortex-M)')
    parser.add_argument('--safety-level', '-s', default='SIL-2',
                       help='Safety integrity level (default: SIL-2)')
    parser.add_argument('--validation-level', default='standard',
                       choices=['basic', 'standard', 'comprehensive', 'certification_ready'],
                       help='Validation level (default: standard)')
    
    # Execution modes
    parser.add_argument('--cli', action='store_true', help='Use CLI interface')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--validate', type=str, help='Validate existing SRS document')
    
    # Utility options
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--version', action='version', version='SRS Agent 2.0')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        if args.debug:
            os.environ['DEBUG'] = 'true'
            os.environ['LOG_LEVEL'] = 'DEBUG'
        elif args.verbose:
            os.environ['LOG_LEVEL'] = 'INFO'
        
        logger = setup_logging()
        
        # Use CLI interface if requested
        if args.cli:
            sys.argv = ['cli.py'] + sys.argv[2:]  # Remove --cli from args
            return cli()
        
        # Validate existing document
        if args.validate:
            return validate_document(args.validate, args.validation_level, logger)
        
        # Generate new SRS
        if args.project and args.input:
            return generate_srs(args, logger)
        
        # Interactive mode
        if args.interactive:
            return interactive_generation(args, logger)
        
        # Show help if no valid options provided
        parser.print_help()
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def generate_srs(args: argparse.Namespace, logger) -> int:
    """Generate SRS document from command line arguments."""
    with log_operation("srs_generation", ComponentType.CORE, logger, 
                      project_name=args.project):
        
        print(f"🚀 Starting SRS generation for: {args.project}")
        print(f"📄 Input files: {', '.join(args.input)}")
        
        # Validate input files
        valid_files = []
        for file_path in args.input:
            if Path(file_path).exists():
                valid_files.append(file_path)
            else:
                print(f"⚠️  Warning: File not found: {file_path}")
        
        if not valid_files:
            print("❌ No valid input files found")
            return 1
        
        try:
            # Initialize agent
            print("🤖 Initializing SRS Generation Agent...")
            agent = SRSGenerationAgent()
            
            # Generate SRS
            print("⚙️  Generating SRS document...")
            result = agent.generate_srs(
                project_name=args.project,
                file_paths=valid_files,
                target_architecture=args.architecture,
                safety_level=args.safety_level
            )
            
            # Validate result
            print("✅ Validating generated SRS...")
            validation_engine = ValidationEngine()
            validation_result = validation_engine.validate_document(
                result["srs_document"], args.validation_level
            )
            
            # Save results
            output_path = args.output or f"{args.project.lower().replace(' ', '_')}_srs.md"
            
            # Save SRS document
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(result["srs_document"])
            
            # Save validation report
            validation_path = output_path.replace('.md', '_validation.md')
            validation_engine.export_validation_report(validation_result, validation_path)
            
            # Show summary
            print("\n🎉 SRS Generation Complete!")
            print(f"📄 SRS Document: {output_path}")
            print(f"📊 Validation Report: {validation_path}")
            print(f"⭐ Validation Score: {validation_result.overall_score:.1f}/100")
            print(f"🎯 Status: {'PASS' if validation_result.passed else 'NEEDS IMPROVEMENT'}")
            
            metadata = result.get("metadata", {})
            print(f"📈 Requirements Generated: {metadata.get('total_requirements', 0)}")
            print(f"📝 Sections: {metadata.get('sections_generated', 0)}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            return 1


def validate_document(doc_path: str, validation_level: str, logger) -> int:
    """Validate an existing SRS document."""
    with log_operation("srs_validation", ComponentType.VALIDATOR, logger,
                      file_path=doc_path):
        
        print(f"🔍 Validating SRS document: {doc_path}")
        
        if not Path(doc_path).exists():
            print(f"❌ File not found: {doc_path}")
            return 1
        
        try:
            # Read document
            with open(doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Validate
            validation_engine = ValidationEngine()
            result = validation_engine.validate_document(content, validation_level)
            
            # Show results
            print(f"\n📊 Validation Results:")
            print(f"⭐ Score: {result.overall_score:.1f}/100")
            print(f"📊 Grade: {result.summary.get('grade', 'N/A')}")
            print(f"🎯 Status: {'PASS' if result.passed else 'FAIL'}")
            print(f"📋 Level: {result.validation_level}")
            
            if result.recommendations:
                print(f"\n💡 Top Recommendations:")
                for i, rec in enumerate(result.recommendations[:3], 1):
                    print(f"  {i}. {rec}")
            
            # Save validation report
            report_path = doc_path.replace('.md', '_validation.md')
            if validation_engine.export_validation_report(result, report_path):
                print(f"\n📄 Detailed report saved: {report_path}")
            
            return 0 if result.passed else 1
            
        except Exception as e:
            print(f"❌ Error during validation: {str(e)}")
            return 1


def interactive_generation(args: argparse.Namespace, logger) -> int:
    """Interactive SRS generation mode."""
    with log_operation("interactive_generation", ComponentType.CLI, logger):
        
        print("🎯 Interactive SRS Generation Mode")
        print("=" * 40)
        
        # Get project information
        project_name = args.project or input("Enter project name: ").strip()
        if not project_name:
            print("❌ Project name is required")
            return 1
        
        # Get input files
        input_files = []
        if args.input:
            input_files = args.input
        else:
            print("\nEnter input file paths (one per line, empty line to finish):")
            while True:
                file_path = input("File path: ").strip()
                if not file_path:
                    break
                if Path(file_path).exists():
                    input_files.append(file_path)
                else:
                    print(f"⚠️  File not found: {file_path}")
        
        if not input_files:
            print("❌ No input files provided")
            return 1
        
        # Get configuration
        print(f"\nCurrent architecture: {args.architecture}")
        new_arch = input("Enter new architecture (or press Enter to keep current): ").strip()
        if new_arch:
            args.architecture = new_arch
        
        print(f"Current safety level: {args.safety_level}")
        new_safety = input("Enter new safety level (or press Enter to keep current): ").strip()
        if new_safety:
            args.safety_level = new_safety
        
        # Confirmation
        print(f"\n📋 Generation Summary:")
        print(f"   Project: {project_name}")
        print(f"   Files: {len(input_files)} documents")
        print(f"   Architecture: {args.architecture}")
        print(f"   Safety Level: {args.safety_level}")
        
        confirm = input("\nProceed with generation? (y/N): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Generation cancelled")
            return 0
        
        # Update args and generate
        args.project = project_name
        args.input = input_files
        
        return generate_srs(args, logger)


def create_example_project():
    """Create example project with sample files."""
    example_dir = Path("example_project")
    example_dir.mkdir(exist_ok=True)
    
    # Create sample requirement document
    sample_req = """# Motor Control System Requirements

## System Overview
The motor control system shall provide precise speed and torque control for automotive applications.

## Functional Requirements
- The system shall control motor speed with ±1% accuracy
- The system shall respond to speed commands within 10ms
- The system shall implement regenerative braking

## Performance Requirements
- Maximum motor speed: 10,000 RPM
- Continuous power output: 50 kW
- Peak power output: 100 kW for 30 seconds

## Safety Requirements
- The system shall implement fail-safe shutdown
- Emergency stop shall activate within 100ms
- System shall comply with ISO 26262 ASIL-D
"""
    
    with open(example_dir / "requirements.md", "w") as f:
        f.write(sample_req)
    
    # Create sample specification
    sample_spec = """# Technical Specification

## Hardware Platform
- Processor: ARM Cortex-M7 at 400MHz
- Memory: 2MB Flash, 512KB RAM
- Communication: CAN-FD, Ethernet
- I/O: 16 analog inputs, 12 PWM outputs

## Software Architecture
- Real-time operating system (FreeRTOS)
- Control algorithm: Field-Oriented Control (FOC)
- Communication stack: AUTOSAR Classic
- Diagnostic protocol: UDS over CAN

## Environmental Requirements
- Operating temperature: -40°C to +125°C
- Vibration resistance: IEC 60068-2-6
- EMC compliance: ISO 11452
"""
    
    with open(example_dir / "specification.md", "w") as f:
        f.write(sample_spec)
    
    print(f"✅ Example project created in: {example_dir}")
    print("📄 Sample files:")
    print(f"   - {example_dir}/requirements.md")
    print(f"   - {example_dir}/specification.md")
    print("\nTo generate SRS:")
    print(f"python run.py --project 'Motor Control Example' --input {example_dir}/*.md")


def show_system_info():
    """Show system information and configuration."""
    print("🔧 SRS Generation Agent - System Information")
    print("=" * 50)
    
    try:
        config = get_config()
        print(f"Configuration loaded: ✅")
        print(f"Default AI Provider: {config.model.default_provider}")
        print(f"Chunk Size: {config.processing.chunk_size}")
        print(f"Supported Formats: {', '.join(config.processing.supported_formats)}")
        print(f"Vector Store: {config.processing.vector_store_type}")
        
        # Check API keys
        if config.model.openai_api_key:
            print("OpenAI API Key: ✅ Configured")
        else:
            print("OpenAI API Key: ❌ Not configured")
            
        if config.model.anthropic_api_key:
            print("Anthropic API Key: ✅ Configured")
        else:
            print("Anthropic API Key: ❌ Not configured")
            
    except Exception as e:
        print(f"Configuration error: {str(e)}")
    
    print(f"\nPython version: {sys.version}")
    print(f"Current directory: {Path.cwd()}")


def run_quick_test():
    """Run a quick system test."""
    print("🧪 Running Quick System Test...")
    
    try:
        # Test configuration
        print("1. Testing configuration... ", end="")
        config = get_config()
        print("✅")
        
        # Test logging
        print("2. Testing logging... ", end="")
        logger = get_logger()
        logger.info("Test log message")
        print("✅")
        
        # Test document parsing
        print("3. Testing document parsers... ", end="")
        from document_parsers import DocumentParserFactory
        supported = DocumentParserFactory.get_supported_extensions()
        print(f"✅ ({len(supported)} formats)")
        
        # Test template system
        print("4. Testing template system... ", end="")
        from srs_template_manager import SRSTemplateManager
        template_mgr = SRSTemplateManager()
        structure = template_mgr.get_document_structure()
        print(f"✅ ({len(structure)} sections)")
        
        print("\n🎉 All tests passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Special commands
    if len(sys.argv) == 2:
        command = sys.argv[1]
        
        if command == "--create-example":
            create_example_project()
            sys.exit(0)
        elif command == "--system-info":
            show_system_info()
            sys.exit(0)
        elif command == "--test":
            success = run_quick_test()
            sys.exit(0 if success else 1)
    
    # Run main application
    exit_code = main()
    sys.exit(exit_code)