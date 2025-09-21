"""
Validation and Quality Check Engine for SRS documents.
Validates requirements, document structure, and compliance with standards.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    score: float  # 0.0 to 100.0
    severity: str  # "error", "warning", "info"
    message: str
    details: List[str]
    suggestions: List[str]


@dataclass
class RequirementValidationResult:
    """Result of requirement validation."""
    requirement_id: str
    passed: bool
    issues: List[ValidationResult]
    score: float


@dataclass
class DocumentValidationResult:
    """Complete document validation result."""
    overall_score: float
    passed: bool
    validation_level: str
    timestamp: str
    summary: Dict[str, Any]
    section_results: Dict[str, List[ValidationResult]]
    requirement_results: List[RequirementValidationResult]
    compliance_results: Dict[str, ValidationResult]
    recommendations: List[str]


class ValidationEngine:
    """Main validation engine for SRS documents."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.validation_rules = None
        self.load_validation_rules()
    
    def load_validation_rules(self):
        """Load validation rules from JSON file."""
        rules_path = self.config.paths.templates_dir / self.config.paths.validation_rules_file
        
        try:
            with open(rules_path, 'r', encoding='utf-8') as file:
                self.validation_rules = json.load(file)
            logger.info(f"Loaded validation rules from {rules_path}")
        except FileNotFoundError:
            logger.error(f"Validation rules file not found: {rules_path}")
            self.validation_rules = self._create_default_rules()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing validation rules JSON: {str(e)}")
            self.validation_rules = self._create_default_rules()
        except Exception as e:
            logger.error(f"Error loading validation rules: {str(e)}")
            self.validation_rules = self._create_default_rules()
    
    def _create_default_rules(self) -> Dict[str, Any]:
        """Create minimal default validation rules."""
        return {
            "validation_rules": {
                "requirement_validation": {
                    "format_rules": {},
                    "content_rules": {},
                    "semiconductor_specific": {}
                },
                "document_validation": {
                    "structure_rules": {},
                    "completeness_rules": {},
                    "consistency_rules": {}
                },
                "quality_metrics": {},
                "compliance_standards": {},
                "validation_levels": {
                    "basic": {"threshold_score": 60},
                    "standard": {"threshold_score": 75},
                    "comprehensive": {"threshold_score": 85},
                    "certification_ready": {"threshold_score": 95}
                }
            }
        }
    
    def validate_document(self, srs_content: str, validation_level: str = "standard",
                         project_context: Dict[str, Any] = None) -> DocumentValidationResult:
        """Validate complete SRS document."""
        logger.info(f"Starting document validation at {validation_level} level")
        
        try:
            # Parse document structure
            document_structure = self._parse_document_structure(srs_content)
            
            # Extract requirements
            requirements = self._extract_requirements_from_content(srs_content)
            
            # Run validation checks
            section_results = self._validate_document_structure(document_structure)
            requirement_results = self._validate_requirements(requirements)
            compliance_results = self._validate_compliance(srs_content, project_context)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                section_results, requirement_results, compliance_results
            )
            
            # Determine pass/fail
            threshold = self._get_validation_threshold(validation_level)
            passed = overall_score >= threshold
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                section_results, requirement_results, compliance_results
            )
            
            return DocumentValidationResult(
                overall_score=overall_score,
                passed=passed,
                validation_level=validation_level,
                timestamp=datetime.now().isoformat(),
                summary=self._create_validation_summary(overall_score, passed, validation_level),
                section_results=section_results,
                requirement_results=requirement_results,
                compliance_results=compliance_results,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error during document validation: {str(e)}")
            return self._create_error_result(str(e), validation_level)
    
    def validate_requirement(self, requirement: Dict[str, Any]) -> RequirementValidationResult:
        """Validate a single requirement."""
        req_id = requirement.get("id", "UNKNOWN")
        issues = []
        
        try:
            # Format validation
            format_issues = self._validate_requirement_format(requirement)
            issues.extend(format_issues)
            
            # Content validation
            content_issues = self._validate_requirement_content(requirement)
            issues.extend(content_issues)
            
            # Semiconductor-specific validation
            semiconductor_issues = self._validate_semiconductor_specific(requirement)
            issues.extend(semiconductor_issues)
            
            # Calculate requirement score
            error_count = sum(1 for issue in issues if issue.severity == "error")
            warning_count = sum(1 for issue in issues if issue.severity == "warning")
            
            score = max(0, 100 - (error_count * 20) - (warning_count * 5))
            passed = error_count == 0
            
            return RequirementValidationResult(
                requirement_id=req_id,
                passed=passed,
                issues=issues,
                score=score
            )
            
        except Exception as e:
            logger.error(f"Error validating requirement {req_id}: {str(e)}")
            return RequirementValidationResult(
                requirement_id=req_id,
                passed=False,
                issues=[ValidationResult(
                    check_name="validation_error",
                    passed=False,
                    score=0.0,
                    severity="error",
                    message=f"Validation error: {str(e)}",
                    details=[],
                    suggestions=[]
                )],
                score=0.0
            )
    
    def _parse_document_structure(self, content: str) -> Dict[str, Any]:
        """Parse document structure from content."""
        structure = {
            "sections": [],
            "total_length": len(content),
            "line_count": len(content.split('\n'))
        }
        
        # Find sections by headers
        section_pattern = r'^#{1,3}\s+(.+)$'
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.match(section_pattern, line)
            if match:
                title = match.group(1).strip()
                level = line.count('#')
                structure["sections"].append({
                    "title": title,
                    "level": level,
                    "line_number": i + 1,
                    "section_id": self._title_to_section_id(title)
                })
        
        return structure
    
    def _extract_requirements_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract requirements from document content."""
        requirements = []
        
        # Look for requirement patterns
        req_patterns = [
            r'([A-Z]{2,5}-[A-Z0-9]+-\d{3}):\s*(.+)',  # ID: Text format
            r'([A-Z]{2,5}-[A-Z0-9]+-\d{3})\s+(.+)',   # ID Text format
            r'(\d+\.\d+\.\d+)\s+(.+shall.+)',          # Numbered requirements
        ]
        
        for pattern in req_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                req_id, req_text = match
                if "shall" in req_text.lower():
                    requirements.append({
                        "id": req_id,
                        "text": req_text.strip(),
                        "priority": self._extract_priority(req_text),
                        "source_line": content.find(f"{req_id}")
                    })
        
        return requirements
    
    def _validate_requirement_format(self, requirement: Dict[str, Any]) -> List[ValidationResult]:
        """Validate requirement format according to rules."""
        issues = []
        format_rules = self.validation_rules["validation_rules"]["requirement_validation"]["format_rules"]
        
        # Validate requirement structure
        req_text = requirement.get("text", "")
        structure_rule = format_rules.get("requirement_structure", {})
        pattern = structure_rule.get("pattern", "")
        
        if pattern and not re.match(pattern, req_text, re.IGNORECASE):
            issues.append(ValidationResult(
                check_name="requirement_structure",
                passed=False,
                score=0.0,
                severity="error",
                message="Requirement does not follow proper structure",
                details=[f"Text: {req_text}", f"Expected pattern: {pattern}"],
                suggestions=["Start requirement with 'The system shall' or similar"]
            ))
        
        # Validate requirement ID
        req_id = requirement.get("id", "")
        id_rule = format_rules.get("requirement_id", {})
        id_pattern = id_rule.get("pattern", "")
        
        if id_pattern and not re.match(id_pattern, req_id):
            issues.append(ValidationResult(
                check_name="requirement_id_format",
                passed=False,
                score=0.0,
                severity="error",
                message="Requirement ID format is invalid",
                details=[f"ID: {req_id}", f"Expected pattern: {id_pattern}"],
                suggestions=["Use format like FR-SYS-001, NFR-PERF-012"]
            ))
        
        # Validate priority
        priority = requirement.get("priority", "")
        allowed_priorities = format_rules.get("priority_levels", {}).get("allowed_values", [])
        
        if allowed_priorities and priority not in allowed_priorities:
            issues.append(ValidationResult(
                check_name="priority_validation",
                passed=False,
                score=50.0,
                severity="error" if not priority else "warning",
                message="Invalid or missing priority level",
                details=[f"Priority: {priority}", f"Allowed: {allowed_priorities}"],
                suggestions=[f"Use one of: {', '.join(allowed_priorities)}"]
            ))
        
        return issues
    
    def _validate_requirement_content(self, requirement: Dict[str, Any]) -> List[ValidationResult]:
        """Validate requirement content quality."""
        issues = []
        content_rules = self.validation_rules["validation_rules"]["requirement_validation"]["content_rules"]
        req_text = requirement.get("text", "").lower()
        
        # Check measurability
        measurability_rule = content_rules.get("measurability", {})
        keywords = measurability_rule.get("keywords", [])
        
        has_measurable_criteria = any(keyword in req_text for keyword in keywords)
        if not has_measurable_criteria:
            issues.append(ValidationResult(
                check_name="measurability",
                passed=False,
                score=70.0,
                severity="warning",
                message="Requirement lacks measurable criteria",
                details=[f"Text: {requirement.get('text', '')}"],
                suggestions=["Add specific, measurable criteria (e.g., 'within 10ms', 'less than 50MB')"]
            ))
        
        # Check testability
        testability_rule = content_rules.get("testability", {})
        prohibited_phrases = testability_rule.get("prohibited_phrases", [])
        
        for phrase in prohibited_phrases:
            if phrase in req_text:
                issues.append(ValidationResult(
                    check_name="testability",
                    passed=False,
                    score=30.0,
                    severity="error",
                    message=f"Requirement contains untestable phrase: '{phrase}'",
                    details=[f"Text: {requirement.get('text', '')}"],
                    suggestions=["Replace with specific, testable criteria"]
                ))
        
        # Check atomicity
        atomicity_rule = content_rules.get("atomicity", {})
        prohibited_words = atomicity_rule.get("prohibited_words", [])
        
        compound_words = [word for word in prohibited_words if word in req_text]
        if compound_words:
            issues.append(ValidationResult(
                check_name="atomicity",
                passed=False,
                score=60.0,
                severity="warning",
                message="Requirement may not be atomic",
                details=[f"Found words: {compound_words}"],
                suggestions=["Split into separate requirements for each function"]
            ))
        
        # Check completeness
        completeness_rule = content_rules.get("completeness", {})
        required_attrs = completeness_rule.get("required_attributes", [])
        
        missing_attrs = [attr for attr in required_attrs if not requirement.get(attr)]
        if missing_attrs:
            issues.append(ValidationResult(
                check_name="completeness",
                passed=False,
                score=40.0,
                severity="error",
                message="Missing required attributes",
                details=[f"Missing: {missing_attrs}"],
                suggestions=[f"Add missing attributes: {', '.join(missing_attrs)}"]
            ))
        
        return issues
    
    def _validate_semiconductor_specific(self, requirement: Dict[str, Any]) -> List[ValidationResult]:
        """Validate semiconductor-specific requirements."""
        issues = []
        semiconductor_rules = self.validation_rules["validation_rules"]["requirement_validation"]["semiconductor_specific"]
        req_text = requirement.get("text", "").lower()
        
        # Check real-time constraints
        if "real" in req_text or "time" in req_text or "latency" in req_text:
            rt_rule = semiconductor_rules.get("real_time_constraints", {})
            timing_units = rt_rule.get("timing_units", [])
            
            has_timing_unit = any(unit in req_text for unit in timing_units)
            if not has_timing_unit:
                issues.append(ValidationResult(
                    check_name="real_time_constraints",
                    passed=False,
                    score=60.0,
                    severity="warning",
                    message="Real-time requirement missing timing units",
                    details=[f"Text: {requirement.get('text', '')}"],
                    suggestions=[f"Specify timing units: {', '.join(timing_units[:5])}"]
                ))
        
        # Check power management
        if "power" in req_text or "energy" in req_text or "consumption" in req_text:
            power_rule = semiconductor_rules.get("power_management", {})
            power_units = power_rule.get("power_units", [])
            
            has_power_unit = any(unit in req_text for unit in power_units)
            if not has_power_unit:
                issues.append(ValidationResult(
                    check_name="power_management",
                    passed=False,
                    score=70.0,
                    severity="warning",
                    message="Power requirement missing units",
                    details=[f"Text: {requirement.get('text', '')}"],
                    suggestions=[f"Specify power units: {', '.join(power_units[:5])}"]
                ))
        
        # Check safety integrity
        if "safety" in req_text or "sil" in req_text:
            safety_rule = semiconductor_rules.get("safety_integrity", {})
            required_attrs = safety_rule.get("required_for_safety", [])
            
            missing_safety_attrs = [attr for attr in required_attrs if not requirement.get(attr)]
            if missing_safety_attrs:
                issues.append(ValidationResult(
                    check_name="safety_integrity",
                    passed=False,
                    score=40.0,
                    severity="error",
                    message="Safety requirement missing required attributes",
                    details=[f"Missing: {missing_safety_attrs}"],
                    suggestions=[f"Add safety attributes: {', '.join(missing_safety_attrs)}"]
                ))
        
        return issues
    
    def _validate_document_structure(self, document_structure: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
        """Validate document structure."""
        results = {}
        structure_rules = self.validation_rules["validation_rules"]["document_validation"]["structure_rules"]
        
        # Check required sections
        required_sections = structure_rules.get("required_sections", [])
        found_sections = [section["section_id"] for section in document_structure.get("sections", [])]
        
        missing_sections = [section for section in required_sections if section not in found_sections]
        
        structure_issues = []
        if missing_sections:
            structure_issues.append(ValidationResult(
                check_name="required_sections",
                passed=False,
                score=50.0,
                severity="error",
                message="Missing required sections",
                details=[f"Missing: {missing_sections}"],
                suggestions=[f"Add sections: {', '.join(missing_sections)}"]
            ))
        else:
            structure_issues.append(ValidationResult(
                check_name="required_sections",
                passed=True,
                score=100.0,
                severity="info",
                message="All required sections present",
                details=[],
                suggestions=[]
            ))
        
        results["document_structure"] = structure_issues
        return results
    
    def _validate_requirements(self, requirements: List[Dict[str, Any]]) -> List[RequirementValidationResult]:
        """Validate all requirements."""
        return [self.validate_requirement(req) for req in requirements]
    
    def _validate_compliance(self, content: str, project_context: Dict[str, Any] = None) -> Dict[str, ValidationResult]:
        """Validate compliance with standards."""
        compliance_results = {}
        
        if not project_context:
            project_context = {}
        
        # Get applicable standards
        standards = project_context.get("compliance_standards", self.config.semiconductor.compliance_standards)
        
        for standard in standards:
            compliance_results[standard] = self._validate_standard_compliance(content, standard)
        
        return compliance_results
    
    def _validate_standard_compliance(self, content: str, standard: str) -> ValidationResult:
        """Validate compliance with a specific standard."""
        content_lower = content.lower()
        
        # Basic compliance checking (would be more sophisticated in practice)
        compliance_keywords = {
            "ISO 26262": ["hazard", "asil", "safety goal", "safety requirement"],
            "IEC 61508": ["sil", "safety function", "safety integrity"],
            "DO-178C": ["dal", "verification", "validation", "software level"],
            "MISRA C": ["coding standard", "static analysis", "rule compliance"]
        }
        
        keywords = compliance_keywords.get(standard, [])
        found_keywords = [kw for kw in keywords if kw in content_lower]
        
        score = (len(found_keywords) / len(keywords) * 100) if keywords else 100
        passed = score >= 50  # At least half the keywords should be present
        
        return ValidationResult(
            check_name=f"compliance_{standard.replace(' ', '_')}",
            passed=passed,
            score=score,
            severity="warning" if not passed else "info",
            message=f"Compliance with {standard}: {'Pass' if passed else 'Needs Review'}",
            details=[f"Found keywords: {found_keywords}", f"Expected keywords: {keywords}"],
            suggestions=["Add more standard-specific content"] if not passed else []
        )
    
    def _calculate_overall_score(self, section_results: Dict[str, List[ValidationResult]],
                               requirement_results: List[RequirementValidationResult],
                               compliance_results: Dict[str, ValidationResult]) -> float:
        """Calculate overall validation score."""
        quality_metrics = self.validation_rules["validation_rules"]["quality_metrics"]
        
        # Calculate component scores
        section_score = self._calculate_section_score(section_results)
        requirement_score = self._calculate_requirement_score(requirement_results)
        compliance_score = self._calculate_compliance_score(compliance_results)
        
        # Apply weights
        completeness_weight = quality_metrics.get("completeness_score", {}).get("weight", 0.25)
        consistency_weight = quality_metrics.get("consistency_score", {}).get("weight", 0.20)
        testability_weight = quality_metrics.get("testability_score", {}).get("weight", 0.20)
        traceability_weight = quality_metrics.get("traceability_score", {}).get("weight", 0.15)
        compliance_weight = quality_metrics.get("compliance_score", {}).get("weight", 0.20)
        
        # Weighted average
        overall_score = (
            section_score * completeness_weight +
            requirement_score * (consistency_weight + testability_weight + traceability_weight) +
            compliance_score * compliance_weight
        )
        
        return min(100.0, max(0.0, overall_score))
    
    def _calculate_section_score(self, section_results: Dict[str, List[ValidationResult]]) -> float:
        """Calculate section validation score."""
        all_results = []
        for results in section_results.values():
            all_results.extend(results)
        
        if not all_results:
            return 100.0
        
        return sum(result.score for result in all_results) / len(all_results)
    
    def _calculate_requirement_score(self, requirement_results: List[RequirementValidationResult]) -> float:
        """Calculate requirement validation score."""
        if not requirement_results:
            return 100.0
        
        return sum(result.score for result in requirement_results) / len(requirement_results)
    
    def _calculate_compliance_score(self, compliance_results: Dict[str, ValidationResult]) -> float:
        """Calculate compliance validation score."""
        if not compliance_results:
            return 100.0
        
        scores = [result.score for result in compliance_results.values()]
        return sum(scores) / len(scores)
    
    def _get_validation_threshold(self, validation_level: str) -> float:
        """Get validation threshold for a specific level."""
        levels = self.validation_rules["validation_rules"]["validation_levels"]
        return levels.get(validation_level, {}).get("threshold_score", 75.0)
    
    def _generate_recommendations(self, section_results: Dict[str, List[ValidationResult]],
                                requirement_results: List[RequirementValidationResult],
                                compliance_results: Dict[str, ValidationResult]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Collect suggestions from all validation results
        for results in section_results.values():
            for result in results:
                recommendations.extend(result.suggestions)
        
        for req_result in requirement_results:
            for issue in req_result.issues:
                recommendations.extend(issue.suggestions)
        
        for comp_result in compliance_results.values():
            recommendations.extend(comp_result.suggestions)
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def _create_validation_summary(self, score: float, passed: bool, level: str) -> Dict[str, Any]:
        """Create validation summary."""
        return {
            "overall_score": score,
            "passed": passed,
            "validation_level": level,
            "grade": self._score_to_grade(score),
            "status": "PASS" if passed else "FAIL",
            "recommendation": self._get_level_recommendation(score)
        }
    
    def _create_error_result(self, error_message: str, validation_level: str) -> DocumentValidationResult:
        """Create error result when validation fails."""
        return DocumentValidationResult(
            overall_score=0.0,
            passed=False,
            validation_level=validation_level,
            timestamp=datetime.now().isoformat(),
            summary={"error": error_message},
            section_results={},
            requirement_results=[],
            compliance_results={},
            recommendations=["Fix validation errors and retry"]
        )
    
    # Utility methods
    def _title_to_section_id(self, title: str) -> str:
        """Convert section title to section ID."""
        # Remove numbers and convert to lowercase with underscores
        clean_title = re.sub(r'^\d+\.?\s*', '', title.lower())
        return re.sub(r'[^a-z0-9]+', '_', clean_title).strip('_')
    
    def _extract_priority(self, text: str) -> str:
        """Extract priority from requirement text."""
        priorities = ["critical", "high", "medium", "low"]
        text_lower = text.lower()
        
        for priority in priorities:
            if priority in text_lower:
                return priority.capitalize()
        
        return "Medium"  # Default priority
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_level_recommendation(self, score: float) -> str:
        """Get recommendation based on score."""
        if score >= 95:
            return "Ready for certification review"
        elif score >= 85:
            return "Suitable for production use"
        elif score >= 75:
            return "Acceptable with minor improvements"
        elif score >= 60:
            return "Requires significant improvements"
        else:
            return "Major revision needed"
    
    def export_validation_report(self, result: DocumentValidationResult, output_path: str) -> bool:
        """Export validation report to file."""
        try:
            report_content = self._generate_validation_report(result)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(report_content)
            
            logger.info(f"Validation report exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting validation report: {str(e)}")
            return False
    
    def _generate_validation_report(self, result: DocumentValidationResult) -> str:
        """Generate formatted validation report."""
        report = f"""# SRS Validation Report

## Summary
- **Overall Score:** {result.overall_score:.1f}/100
- **Status:** {result.summary.get('status', 'UNKNOWN')}
- **Grade:** {result.summary.get('grade', 'N/A')}
- **Validation Level:** {result.validation_level}
- **Generated:** {result.timestamp}

## Validation Results

### Document Structure
"""
        
        for section_name, section_results in result.section_results.items():
            report += f"\n#### {section_name}\n"
            for validation_result in section_results:
                status = "✅ PASS" if validation_result.passed else "❌ FAIL"
                report += f"- {validation_result.check_name}: {status} (Score: {validation_result.score:.1f})\n"
                if validation_result.message:
                    report += f"  - {validation_result.message}\n"
        
        report += "\n### Requirements Validation\n"
        passed_reqs = sum(1 for r in result.requirement_results if r.passed)
        total_reqs = len(result.requirement_results)
        report += f"- **Requirements Passed:** {passed_reqs}/{total_reqs}\n\n"
        
        for req_result in result.requirement_results[:10]:  # Show first 10
            status = "✅" if req_result.passed else "❌"
            report += f"{status} {req_result.requirement_id} (Score: {req_result.score:.1f})\n"
        
        report += "\n### Compliance Results\n"
        for standard, comp_result in result.compliance_results.items():
            status = "✅ PASS" if comp_result.passed else "❌ NEEDS REVIEW"
            report += f"- {standard}: {status} (Score: {comp_result.score:.1f})\n"
        
        report += "\n### Recommendations\n"
        for i, recommendation in enumerate(result.recommendations[:10], 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"\n---\n*Report generated by SRS Validation Engine*\n"
        
        return report


# Utility functions
def create_validation_engine(config=None) -> ValidationEngine:
    """Create and initialize validation engine."""
    return ValidationEngine(config)


def validate_srs_document(srs_content: str, validation_level: str = "standard",
                         project_context: Dict[str, Any] = None,
                         config=None) -> DocumentValidationResult:
    """Convenience function to validate SRS document."""
    engine = ValidationEngine(config)
    return engine.validate_document(srs_content, validation_level, project_context)