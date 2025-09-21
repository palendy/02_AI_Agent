"""
Command Line Interface for SRS Generation Agent.
Provides user-friendly CLI for generating SRS documents.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich import print as rprint

# Local imports
from config import get_config, reload_config
from srs_generation_agent import SRSGenerationAgent
from validation_engine import ValidationEngine
from document_parsers import DocumentParserFactory, get_document_info

# Initialize rich console
console = Console()

# Configure logging for CLI
def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI interface."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('srs_agent.log')
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', type=click.Path(exists=True), help='Custom config file path')
@click.pass_context
def cli(ctx, verbose, config_file):
    """SRS Generation Agent - Generate Software Requirements Specifications for semiconductor firmware."""
    setup_cli_logging(verbose)
    
    # Ensure context object
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config_file'] = config_file
    
    # Show welcome message
    if ctx.invoked_subcommand is None:
        show_welcome()


def show_welcome():
    """Show welcome message and help."""
    welcome_text = Text()
    welcome_text.append("SRS Generation Agent", style="bold blue")
    welcome_text.append("\n\nGenerate comprehensive Software Requirements Specifications\n")
    welcome_text.append("for semiconductor firmware development.\n\n")
    welcome_text.append("Key Features:\n", style="bold")
    welcome_text.append("• Multi-format document processing (PDF, DOCX, TXT, MD)\n")
    welcome_text.append("• Semiconductor-specific SRS templates\n")
    welcome_text.append("• Industry standards compliance (ISO 26262, IEC 61508, etc.)\n")
    welcome_text.append("• Automated validation and quality checks\n")
    welcome_text.append("• LangChain-powered AI generation\n\n")
    welcome_text.append("Use --help with any command for detailed information.", style="italic")
    
    console.print(Panel(welcome_text, title="Welcome", border_style="blue"))


@cli.command()
@click.argument('project_name')
@click.argument('input_files', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--architecture', '-a', default='ARM Cortex-M', help='Target architecture')
@click.option('--safety-level', '-s', default='SIL-2', help='Safety integrity level')
@click.option('--validation-level', default='standard', 
              type=click.Choice(['basic', 'standard', 'comprehensive', 'certification_ready']),
              help='Validation level')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.pass_context
def generate(ctx, project_name, input_files, output, architecture, safety_level, validation_level, interactive):
    """Generate SRS document from input files."""
    try:
        # Interactive setup if requested
        if interactive:
            project_name, architecture, safety_level, validation_level = interactive_setup(
                project_name, architecture, safety_level, validation_level
            )
        
        # Validate input files
        file_paths = list(input_files)
        valid_files = validate_input_files(file_paths)
        
        if not valid_files:
            console.print("[red]Error: No valid input files provided[/red]")
            sys.exit(1)
        
        # Show generation info
        show_generation_info(project_name, valid_files, architecture, safety_level, validation_level)
        
        if not Confirm.ask("Proceed with SRS generation?"):
            console.print("Generation cancelled.")
            return
        
        # Initialize agent
        console.print("\n[blue]Initializing SRS Generation Agent...[/blue]")
        try:
            config = get_config()
            agent = SRSGenerationAgent(config)
        except Exception as e:
            console.print(f"[red]Error initializing agent: {str(e)}[/red]")
            sys.exit(1)
        
        # Generate SRS
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating SRS document...", total=None)
            
            try:
                result = agent.generate_srs(
                    project_name=project_name,
                    file_paths=valid_files,
                    target_architecture=architecture,
                    safety_level=safety_level
                )
            except Exception as e:
                progress.stop()
                console.print(f"[red]Error during generation: {str(e)}[/red]")
                sys.exit(1)
            
            progress.update(task, description="Validating generated SRS...")
            
            # Validate the result
            validation_engine = ValidationEngine(config)
            validation_result = validation_engine.validate_document(
                result["srs_document"], validation_level
            )
            
            progress.update(task, description="Saving results...")
        
        # Show results
        show_generation_results(result, validation_result)
        
        # Save files
        output_path = output or f"{project_name.lower().replace(' ', '_')}_srs.md"
        if save_results(result, validation_result, output_path):
            console.print(f"\n[green]✅ SRS generated successfully![/green]")
            console.print(f"📄 SRS document: {output_path}")
            console.print(f"📊 Validation report: {output_path.replace('.md', '_validation.md')}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.argument('srs_file', type=click.Path(exists=True))
@click.option('--level', default='standard', 
              type=click.Choice(['basic', 'standard', 'comprehensive', 'certification_ready']),
              help='Validation level')
@click.option('--output', '-o', type=click.Path(), help='Validation report output file')
@click.option('--show-details', is_flag=True, help='Show detailed validation results')
def validate(srs_file, level, output, show_details):
    """Validate an existing SRS document."""
    try:
        console.print(f"[blue]Validating SRS document: {srs_file}[/blue]")
        
        # Read SRS content
        with open(srs_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Initialize validation engine
        validation_engine = ValidationEngine()
        
        # Perform validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating document...", total=None)
            result = validation_engine.validate_document(content, level)
        
        # Show validation results
        show_validation_results(result, show_details)
        
        # Save validation report
        if output:
            if validation_engine.export_validation_report(result, output):
                console.print(f"\n[green]Validation report saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during validation: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('files', nargs=-1, required=True, type=click.Path(exists=True))
def analyze(files):
    """Analyze input files for SRS generation readiness."""
    console.print("[blue]Analyzing input files...[/blue]\n")
    
    table = Table(title="File Analysis Results")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Notes")
    
    for file_path in files:
        info = get_document_info(file_path)
        
        # Determine status
        if info.get("error"):
            status = "[red]Error[/red]"
            notes = info["error"]
        elif not info.get("supported"):
            status = "[yellow]Unsupported[/yellow]"
            notes = "File type not supported"
        elif info.get("size_mb", 0) > 50:
            status = "[yellow]Large[/yellow]"
            notes = "File is very large"
        else:
            status = "[green]Ready[/green]"
            notes = "Ready for processing"
        
        table.add_row(
            Path(file_path).name,
            info.get("type", "Unknown"),
            f"{info.get('size_mb', 0):.1f} MB",
            status,
            notes
        )
    
    console.print(table)
    
    # Show supported formats
    supported_formats = DocumentParserFactory.get_supported_extensions()
    console.print(f"\n[blue]Supported formats:[/blue] {', '.join(supported_formats)}")


@cli.command()
@click.option('--show-all', is_flag=True, help='Show all configuration values')
def config(show_all):
    """Show current configuration."""
    try:
        config_obj = get_config()
        
        console.print("[blue]SRS Agent Configuration[/blue]\n")
        
        # Show key configuration
        config_table = Table(title="Configuration Summary")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Default Provider", config_obj.model.default_provider)
        config_table.add_row("Model", 
                           config_obj.model.openai_model if config_obj.model.default_provider == "openai" 
                           else config_obj.model.anthropic_model)
        config_table.add_row("Temperature", str(config_obj.model.temperature))
        config_table.add_row("Max Tokens", str(config_obj.model.max_tokens))
        config_table.add_row("Chunk Size", str(config_obj.processing.chunk_size))
        config_table.add_row("Vector Store", config_obj.processing.vector_store_type)
        config_table.add_row("Log Level", config_obj.logging.log_level)
        
        console.print(config_table)
        
        if show_all:
            console.print("\n[blue]Full Configuration:[/blue]")
            config_dict = config_obj.to_dict()
            console.print_json(data=config_dict)
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")


@cli.command()
@click.option('--template-only', is_flag=True, help='Generate template without content')
@click.option('--output', '-o', type=click.Path(), help='Output template file')
def template(template_only, output):
    """Generate or view SRS template."""
    try:
        from srs_template_manager import SRSTemplateManager
        
        manager = SRSTemplateManager()
        
        if template_only:
            # Generate empty template
            template_content = manager.create_markdown_template()
        else:
            # Show template structure
            structure = manager.generate_complete_document_structure()
            
            console.print("[blue]SRS Template Structure[/blue]\n")
            
            for section in structure["sections"]:
                console.print(f"[bold]{section['title']}[/bold]")
                for subsection in section.get("subsections", []):
                    console.print(f"  • {subsection['title']}")
                    console.print(f"    {subsection.get('description', '')}")
                console.print()
            
            return
        
        # Save template if requested
        if output:
            with open(output, 'w', encoding='utf-8') as file:
                file.write(template_content)
            console.print(f"[green]Template saved to: {output}[/green]")
        else:
            console.print(template_content)
        
    except Exception as e:
        console.print(f"[red]Error generating template: {str(e)}[/red]")


def interactive_setup(project_name, architecture, safety_level, validation_level):
    """Interactive setup for SRS generation."""
    console.print("[blue]🎯 Interactive SRS Generation Setup[/blue]\n")
    
    # Project name
    project_name = Prompt.ask("Project name", default=project_name)
    
    # Architecture
    arch_choices = ["ARM Cortex-M", "RISC-V", "x86", "DSP", "FPGA", "Custom"]
    architecture = Prompt.ask("Target architecture", 
                            choices=arch_choices, default=architecture)
    
    # Safety level
    safety_choices = ["SIL-1", "SIL-2", "SIL-3", "SIL-4", "Non-safety"]
    safety_level = Prompt.ask("Safety integrity level", 
                            choices=safety_choices, default=safety_level)
    
    # Validation level
    validation_choices = ["basic", "standard", "comprehensive", "certification_ready"]
    validation_level = Prompt.ask("Validation level", 
                                choices=validation_choices, default=validation_level)
    
    console.print("\n[green]✅ Setup complete![/green]\n")
    
    return project_name, architecture, safety_level, validation_level


def validate_input_files(file_paths: List[str]) -> List[str]:
    """Validate and filter input files."""
    valid_files = []
    
    for file_path in file_paths:
        if DocumentParserFactory.is_supported(file_path):
            valid_files.append(file_path)
        else:
            console.print(f"[yellow]Warning: Unsupported file format: {file_path}[/yellow]")
    
    return valid_files


def show_generation_info(project_name, files, architecture, safety_level, validation_level):
    """Show generation information before proceeding."""
    info_table = Table(title="Generation Configuration")
    info_table.add_column("Parameter", style="cyan")
    info_table.add_column("Value", style="green")
    
    info_table.add_row("Project Name", project_name)
    info_table.add_row("Target Architecture", architecture)
    info_table.add_row("Safety Level", safety_level)
    info_table.add_row("Validation Level", validation_level)
    info_table.add_row("Input Files", str(len(files)))
    
    console.print("\n", info_table, "\n")
    
    # Show input files
    files_table = Table(title="Input Files")
    files_table.add_column("File", style="cyan")
    files_table.add_column("Size", justify="right")
    
    for file_path in files:
        info = get_document_info(file_path)
        files_table.add_row(
            Path(file_path).name,
            f"{info.get('size_mb', 0):.1f} MB"
        )
    
    console.print(files_table, "\n")


def show_generation_results(result: Dict[str, Any], validation_result):
    """Show generation results."""
    console.print("\n[blue]📊 Generation Results[/blue]\n")
    
    # Generation metadata
    metadata = result.get("metadata", {})
    
    results_table = Table(title="Generation Summary")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Sections Generated", str(metadata.get("sections_generated", 0)))
    results_table.add_row("Total Requirements", str(metadata.get("total_requirements", 0)))
    results_table.add_row("Input Files Processed", str(len(metadata.get("input_files", []))))
    results_table.add_row("Validation Score", f"{validation_result.overall_score:.1f}/100")
    results_table.add_row("Validation Status", 
                        "[green]PASS[/green]" if validation_result.passed else "[red]FAIL[/red]")
    
    console.print(results_table)


def show_validation_results(result, show_details: bool = False):
    """Show validation results."""
    console.print("\n[blue]📋 Validation Results[/blue]\n")
    
    # Summary
    summary = result.summary
    status_color = "green" if result.passed else "red"
    status_text = "PASS" if result.passed else "FAIL"
    
    summary_panel = Panel(
        f"Score: {result.overall_score:.1f}/100\n"
        f"Status: [{status_color}]{status_text}[/{status_color}]\n"
        f"Grade: {summary.get('grade', 'N/A')}\n"
        f"Level: {result.validation_level}",
        title="Validation Summary",
        border_style=status_color
    )
    console.print(summary_panel)
    
    if show_details:
        # Show detailed results
        console.print("\n[blue]Detailed Results:[/blue]\n")
        
        # Section results
        if result.section_results:
            for section_name, section_results in result.section_results.items():
                console.print(f"[bold]{section_name}:[/bold]")
                for validation_result in section_results:
                    status = "✅" if validation_result.passed else "❌"
                    console.print(f"  {status} {validation_result.check_name}: {validation_result.message}")
        
        # Requirements results summary
        if result.requirement_results:
            passed_reqs = sum(1 for r in result.requirement_results if r.passed)
            total_reqs = len(result.requirement_results)
            console.print(f"\n[bold]Requirements:[/bold] {passed_reqs}/{total_reqs} passed")
    
    # Show recommendations
    if result.recommendations:
        console.print("\n[blue]💡 Recommendations:[/blue]")
        for i, recommendation in enumerate(result.recommendations[:5], 1):
            console.print(f"{i}. {recommendation}")


def save_results(result: Dict[str, Any], validation_result, output_path: str) -> bool:
    """Save generation and validation results."""
    try:
        # Save SRS document
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(result["srs_document"])
        
        # Save validation report
        validation_path = output_path.replace('.md', '_validation.md')
        validation_engine = ValidationEngine()
        validation_engine.export_validation_report(validation_result, validation_path)
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error saving results: {str(e)}[/red]")
        return False


if __name__ == '__main__':
    cli()