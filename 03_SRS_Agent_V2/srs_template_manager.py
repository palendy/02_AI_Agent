"""
SRS Template Manager for semiconductor firmware development.
Manages SRS document templates, formatting, and structure generation.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from jinja2 import Template, Environment, FileSystemLoader

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RequirementTemplate:
    """Template for individual requirements."""
    id_prefix: str
    format_template: str
    attributes: List[str]
    category: str


@dataclass
class SectionTemplate:
    """Template for SRS document sections."""
    section_id: str
    title: str
    subsections: Dict[str, Dict[str, Any]]
    required_elements: List[str]


class SRSTemplateManager:
    """Manager for SRS document templates and formatting."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.template_data = None
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.config.paths.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.load_template()
    
    def load_template(self):
        """Load SRS template from JSON file."""
        template_path = self.config.paths.templates_dir / self.config.paths.srs_template_file
        
        try:
            with open(template_path, 'r', encoding='utf-8') as file:
                self.template_data = json.load(file)
            logger.info(f"Loaded SRS template from {template_path}")
        except FileNotFoundError:
            logger.error(f"SRS template file not found: {template_path}")
            self.template_data = self._create_default_template()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing SRS template JSON: {str(e)}")
            self.template_data = self._create_default_template()
        except Exception as e:
            logger.error(f"Error loading SRS template: {str(e)}")
            self.template_data = self._create_default_template()
    
    def _create_default_template(self) -> Dict[str, Any]:
        """Create a minimal default template if loading fails."""
        return {
            "srs_template": {
                "metadata": {
                    "template_version": "1.0",
                    "industry_focus": "semiconductor_firmware"
                },
                "document_structure": {},
                "requirement_templates": {},
                "quality_criteria": {},
                "semiconductor_specific": {}
            }
        }
    
    def get_document_structure(self) -> Dict[str, Any]:
        """Get the document structure from template."""
        return self.template_data.get("srs_template", {}).get("document_structure", {})
    
    def get_section_template(self, section_id: str) -> Optional[SectionTemplate]:
        """Get template for a specific section."""
        structure = self.get_document_structure()
        
        for section_num, section_data in structure.items():
            if section_data.get("section_id") == section_id:
                return SectionTemplate(
                    section_id=section_data["section_id"],
                    title=section_data["title"],
                    subsections=section_data.get("subsections", {}),
                    required_elements=section_data.get("required_elements", [])
                )
        return None
    
    def get_requirement_template(self, req_type: str) -> Optional[RequirementTemplate]:
        """Get template for a specific requirement type."""
        req_templates = self.template_data.get("srs_template", {}).get("requirement_templates", {})
        
        if req_type in req_templates:
            template_data = req_templates[req_type]
            return RequirementTemplate(
                id_prefix=template_data.get("id_prefix", "REQ"),
                format_template=template_data.get("format", ""),
                attributes=template_data.get("attributes", []),
                category=req_type
            )
        return None
    
    def get_quality_criteria(self) -> Dict[str, Any]:
        """Get quality criteria for SRS validation."""
        return self.template_data.get("srs_template", {}).get("quality_criteria", {})
    
    def get_semiconductor_specific_data(self) -> Dict[str, Any]:
        """Get semiconductor-specific template data."""
        return self.template_data.get("srs_template", {}).get("semiconductor_specific", {})
    
    def format_requirement(self, req_type: str, requirement_data: Dict[str, Any]) -> str:
        """Format a requirement using the appropriate template."""
        template = self.get_requirement_template(req_type)
        if not template:
            return f"[Unknown requirement type: {req_type}] {requirement_data.get('text', '')}"
        
        try:
            # Create Jinja2 template from format string
            jinja_template = Template(template.format_template)
            
            # Prepare template variables
            template_vars = {
                "id": requirement_data.get("id", "REQ-XXX"),
                "priority": requirement_data.get("priority", "Medium"),
                **requirement_data
            }
            
            return jinja_template.render(**template_vars)
        except Exception as e:
            logger.error(f"Error formatting requirement: {str(e)}")
            return f"{requirement_data.get('id', 'REQ-XXX')}: {requirement_data.get('text', '')}"
    
    def generate_section_outline(self, section_id: str) -> Dict[str, Any]:
        """Generate outline for a specific section."""
        section_template = self.get_section_template(section_id)
        if not section_template:
            return {"error": f"Section template not found: {section_id}"}
        
        outline = {
            "section_id": section_template.section_id,
            "title": section_template.title,
            "subsections": []
        }
        
        for subsection_id, subsection_data in section_template.subsections.items():
            subsection_outline = {
                "subsection_id": subsection_id,
                "title": subsection_data.get("title", ""),
                "description": subsection_data.get("description", ""),
                "required_elements": subsection_data.get("required_elements", []),
                "placeholder_requirements": self._generate_placeholder_requirements(
                    section_template.section_id, subsection_data
                )
            }
            outline["subsections"].append(subsection_outline)
        
        return outline
    
    def _generate_placeholder_requirements(self, section_id: str, subsection_data: Dict[str, Any]) -> List[str]:
        """Generate placeholder requirements for a subsection."""
        placeholders = []
        required_elements = subsection_data.get("required_elements", [])
        
        # Map section to requirement type
        req_type_mapping = {
            "functional_requirements": "functional",
            "non_functional_requirements": "non_functional",
            "interface_requirements": "interface",
            "safety_security_requirements": "safety"
        }
        
        req_type = req_type_mapping.get(section_id, "functional")
        template = self.get_requirement_template(req_type)
        
        if template:
            for i, element in enumerate(required_elements[:5]):  # Limit to 5 placeholders
                placeholder_data = {
                    "id": f"{template.id_prefix}-{section_id.upper()}-{i+1:03d}",
                    "text": f"Requirements related to {element}",
                    "priority": "Medium",
                    "element": element
                }
                formatted_req = self.format_requirement(req_type, placeholder_data)
                placeholders.append(formatted_req)
        
        return placeholders
    
    def generate_complete_document_structure(self) -> Dict[str, Any]:
        """Generate complete SRS document structure with outlines."""
        document_structure = self.get_document_structure()
        complete_structure = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "template_version": self.template_data.get("srs_template", {}).get("metadata", {}).get("template_version", "1.0"),
                "industry_focus": "semiconductor_firmware"
            },
            "sections": []
        }
        
        for section_num, section_data in document_structure.items():
            section_id = section_data.get("section_id")
            if section_id:
                section_outline = self.generate_section_outline(section_id)
                section_outline["section_number"] = section_num
                complete_structure["sections"].append(section_outline)
        
        return complete_structure
    
    def validate_requirement_structure(self, requirement: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate requirement against template structure."""
        errors = []
        
        # Check required fields
        required_fields = ["id", "text", "priority"]
        for field in required_fields:
            if field not in requirement or not requirement[field]:
                errors.append(f"Missing required field: {field}")
        
        # Validate priority
        valid_priorities = ["Critical", "High", "Medium", "Low"]
        priority = requirement.get("priority")
        if priority and priority not in valid_priorities:
            errors.append(f"Invalid priority: {priority}. Must be one of {valid_priorities}")
        
        # Validate ID format
        req_id = requirement.get("id", "")
        if req_id and not re.match(r'^[A-Z]{1,5}-[A-Z0-9]+-\d{3}$', req_id):
            errors.append(f"Invalid ID format: {req_id}. Expected format: PREFIX-SECTION-###")
        
        return len(errors) == 0, errors
    
    def get_section_requirements_count(self, section_id: str) -> Dict[str, int]:
        """Get expected requirements count for a section."""
        quality_criteria = self.get_quality_criteria()
        completeness = quality_criteria.get("completeness", {})
        
        return {
            "min_requirements": completeness.get("min_requirements_per_section", 3),
            "recommended_requirements": completeness.get("min_requirements_per_section", 3) * 2,
            "required_attributes": len(completeness.get("required_attributes", []))
        }
    
    def create_markdown_template(self, project_name: str = "Semiconductor Firmware System") -> str:
        """Create a markdown SRS template."""
        structure = self.generate_complete_document_structure()
        
        markdown_template = f"""# Software Requirements Specification
## {project_name}

**Document Information:**
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Template Version: {structure['metadata']['template_version']}
- Industry Focus: {structure['metadata']['industry_focus']}

---

"""
        
        for section in structure['sections']:
            section_num = section.get('section_number', '')
            markdown_template += f"## {section['title']}\n\n"
            
            for subsection in section.get('subsections', []):
                markdown_template += f"### {subsection['title']}\n\n"
                markdown_template += f"{subsection.get('description', '')}\n\n"
                
                if subsection.get('required_elements'):
                    markdown_template += "**Required Elements:**\n"
                    for element in subsection['required_elements']:
                        markdown_template += f"- {element}\n"
                    markdown_template += "\n"
                
                if subsection.get('placeholder_requirements'):
                    markdown_template += "**Requirements:**\n"
                    for req in subsection['placeholder_requirements']:
                        markdown_template += f"- {req}\n"
                    markdown_template += "\n"
                
                markdown_template += "---\n\n"
        
        return markdown_template
    
    def get_compliance_standards(self) -> List[str]:
        """Get applicable compliance standards for semiconductor firmware."""
        metadata = self.template_data.get("srs_template", {}).get("metadata", {})
        return metadata.get("compliance_standards", [])
    
    def get_target_architectures(self) -> List[str]:
        """Get supported target architectures."""
        semiconductor_data = self.get_semiconductor_specific_data()
        return self.config.semiconductor.target_architectures
    
    def export_template_to_file(self, output_path: str, format_type: str = "json"):
        """Export current template to file."""
        output_file = Path(output_path)
        
        try:
            if format_type.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(self.template_data, file, indent=2, ensure_ascii=False)
            elif format_type.lower() == "markdown":
                markdown_content = self.create_markdown_template()
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(markdown_content)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            logger.info(f"Template exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting template: {str(e)}")
            return False


# Import regex for validation
import re


class SemiconductorSRSGenerator:
    """Specialized SRS generator for semiconductor firmware."""
    
    def __init__(self, template_manager: SRSTemplateManager = None, config=None):
        self.config = config or get_config()
        self.template_manager = template_manager or SRSTemplateManager(config)
        self.semiconductor_data = self.template_manager.get_semiconductor_specific_data()
    
    def generate_power_management_requirements(self) -> List[Dict[str, Any]]:
        """Generate power management specific requirements."""
        requirements = []
        power_data = self.semiconductor_data.get("power_management", {})
        
        # Low power modes
        for i, mode in enumerate(power_data.get("low_power_modes", [])):
            req = {
                "id": f"PWR-001-{i+1:03d}",
                "text": f"The system shall support {mode} mode with appropriate wake-up mechanisms",
                "priority": "High",
                "category": "power_management",
                "verification_method": "Testing",
                "rationale": f"Power efficiency requirement for {mode} operation"
            }
            requirements.append(req)
        
        return requirements
    
    def generate_safety_requirements(self, sil_level: str = "SIL-2") -> List[Dict[str, Any]]:
        """Generate safety requirements based on SIL level."""
        requirements = []
        safety_standards = self.semiconductor_data.get("safety_standards", {})
        
        base_requirements = [
            {
                "text": "implement watchdog timer with configurable timeout",
                "rationale": "System monitoring and fault detection"
            },
            {
                "text": "perform memory integrity checks using CRC or ECC",
                "rationale": "Data integrity assurance"
            },
            {
                "text": "implement fail-safe state transitions",
                "rationale": "Safe system behavior under fault conditions"
            },
            {
                "text": "provide diagnostic self-test capabilities",
                "rationale": "Fault detection and system health monitoring"
            }
        ]
        
        for i, req_data in enumerate(base_requirements):
            req = {
                "id": f"SAF-001-{i+1:03d}",
                "text": f"The system shall {req_data['text']}",
                "priority": "Critical",
                "category": "safety",
                "sil_level": sil_level,
                "verification_method": "Testing and Analysis",
                "rationale": req_data["rationale"]
            }
            requirements.append(req)
        
        return requirements
    
    def generate_communication_requirements(self, protocols: List[str]) -> List[Dict[str, Any]]:
        """Generate communication protocol requirements."""
        requirements = []
        
        for i, protocol in enumerate(protocols):
            req = {
                "id": f"COM-001-{i+1:03d}",
                "text": f"The system shall support {protocol} communication with error detection and recovery",
                "priority": "High",
                "category": "interface",
                "protocol": protocol,
                "verification_method": "Protocol Testing",
                "rationale": f"Communication requirement for {protocol} interface"
            }
            requirements.append(req)
        
        return requirements
    
    def generate_real_time_requirements(self) -> List[Dict[str, Any]]:
        """Generate real-time constraint requirements."""
        requirements = []
        rt_data = self.semiconductor_data.get("real_time_constraints", {})
        
        base_requirements = [
            {
                "text": "respond to critical interrupts within 10 microseconds",
                "timing": "hard_realtime",
                "priority": "Critical"
            },
            {
                "text": "complete periodic control tasks within their deadline",
                "timing": "hard_realtime", 
                "priority": "Critical"
            },
            {
                "text": "maintain deterministic task scheduling",
                "timing": "deterministic",
                "priority": "High"
            },
            {
                "text": "limit interrupt jitter to less than 1 microsecond",
                "timing": "bounded_jitter",
                "priority": "High"
            }
        ]
        
        for i, req_data in enumerate(base_requirements):
            req = {
                "id": f"RT-001-{i+1:03d}",
                "text": f"The system shall {req_data['text']}",
                "priority": req_data["priority"],
                "category": "non_functional",
                "timing_constraint": req_data["timing"],
                "verification_method": "Timing Analysis",
                "rationale": "Real-time performance requirement"
            }
            requirements.append(req)
        
        return requirements


# Utility functions for template management
def load_custom_template(template_path: str) -> SRSTemplateManager:
    """Load a custom SRS template from file."""
    manager = SRSTemplateManager()
    
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            manager.template_data = json.load(file)
        logger.info(f"Custom template loaded from {template_path}")
    except Exception as e:
        logger.error(f"Error loading custom template: {str(e)}")
        raise
    
    return manager


def create_project_template(project_name: str, target_architecture: str, 
                          safety_level: str = "SIL-2") -> Dict[str, Any]:
    """Create a project-specific template configuration."""
    return {
        "project": {
            "name": project_name,
            "target_architecture": target_architecture,
            "safety_level": safety_level,
            "created_at": datetime.now().isoformat()
        },
        "customizations": {
            "additional_sections": [],
            "modified_requirements": [],
            "compliance_standards": []
        }
    }