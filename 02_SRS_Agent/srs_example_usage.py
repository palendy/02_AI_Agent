"""
SRS Generation Agent - Example Usage and Testing
==============================================

This file demonstrates how to use the SRS Generation Agent with sample
specification documents and provides testing capabilities.

Based on 99_RAG_Note course patterns for practical agent implementation.
"""

import os
import tempfile
from typing import List
from srs_generation_agent import SRSGenerationAgent
from config import ConfigProfiles, AgentConfig


def create_sample_spec_documents() -> List[str]:
    """Create sample specification documents for testing"""
    
    # Sample specification document 1: E-commerce System
    spec1_content = """
    E-Commerce Platform Requirements Specification
    
    1. PROJECT OVERVIEW
    The E-Commerce Platform is a comprehensive online retail solution that enables 
    businesses to sell products and services online. The system will support B2C 
    and B2B transactions with advanced features for inventory management, customer 
    relationship management, and payment processing.
    
    2. STAKEHOLDERS
    - End customers (buyers)
    - Merchants and sellers
    - System administrators
    - Payment processors
    - Shipping providers
    - Marketing team
    
    3. FUNCTIONAL REQUIREMENTS
    
    3.1 User Management
    - User registration and authentication
    - Profile management
    - Role-based access control
    - Password reset functionality
    
    3.2 Product Management
    - Product catalog with categories
    - Product search and filtering
    - Inventory tracking
    - Product reviews and ratings
    - Wishlist functionality
    
    3.3 Shopping Cart and Checkout
    - Add/remove items from cart
    - Cart persistence across sessions
    - Multiple payment methods support
    - Order confirmation and tracking
    - Invoice generation
    
    3.4 Payment Processing
    - Credit/debit card processing
    - Digital wallet integration (PayPal, Apple Pay)
    - Secure payment gateway integration
    - Refund and return processing
    
    3.5 Order Management
    - Order history and tracking
    - Order status updates
    - Shipping integration
    - Return/exchange processing
    
    4. NON-FUNCTIONAL REQUIREMENTS
    
    4.1 Performance
    - Page load time < 3 seconds
    - Support 10,000 concurrent users
    - 99.9% uptime availability
    - Response time < 2 seconds for search queries
    
    4.2 Security
    - PCI DSS compliance for payment processing
    - SSL/TLS encryption for all transactions
    - Regular security audits
    - Data protection and privacy compliance (GDPR)
    
    4.3 Scalability
    - Horizontal scaling capability
    - Auto-scaling based on traffic
    - Database replication for high availability
    
    4.4 Usability
    - Mobile-responsive design
    - Accessibility compliance (WCAG 2.1)
    - Multi-language support
    - Intuitive user interface
    
    5. SYSTEM INTERFACES
    
    5.1 External APIs
    - Payment gateway APIs (Stripe, PayPal)
    - Shipping provider APIs (UPS, FedEx, DHL)
    - Tax calculation services
    - Email service providers
    
    5.2 Database Integration
    - PostgreSQL for transactional data
    - Redis for caching and session management
    - Elasticsearch for product search
    
    5.3 Third-party Services
    - CDN for static content delivery
    - Analytics services (Google Analytics)
    - Customer support chat integration
    
    6. DATA REQUIREMENTS
    
    6.1 Data Storage
    - Customer data with privacy protection
    - Product catalog with multimedia content
    - Order and transaction history
    - Inventory and stock management data
    
    6.2 Data Security
    - Encryption at rest and in transit
    - Regular automated backups
    - Data retention policies
    - Audit trails for all transactions
    
    6.3 Data Migration
    - Import existing customer data
    - Product catalog migration
    - Historical order data transfer
    
    7. INTEGRATION REQUIREMENTS
    - ERP system integration for inventory
    - CRM system for customer management
    - Marketing automation platforms
    - Business intelligence and reporting tools
    
    8. CONSTRAINTS AND ASSUMPTIONS
    
    8.1 Technical Constraints
    - Must be cloud-native (AWS/Azure)
    - Microservices architecture required
    - API-first design approach
    
    8.2 Business Constraints
    - Budget limit of $500,000
    - Go-live date within 8 months
    - Must support existing business processes
    
    8.3 Assumptions
    - Users have modern web browsers
    - Reliable internet connectivity
    - Third-party services availability
    """
    
    # Sample specification document 2: Healthcare Management System
    spec2_content = """
    Healthcare Management System Requirements
    
    1. SYSTEM OVERVIEW
    The Healthcare Management System (HMS) is a comprehensive solution for hospitals 
    and healthcare facilities to manage patient care, medical records, appointments, 
    billing, and administrative operations.
    
    2. KEY STAKEHOLDERS
    - Patients and their families
    - Doctors and medical staff
    - Nurses and healthcare providers
    - Administrative staff
    - Insurance companies
    - Pharmacy and laboratory services
    - Hospital management
    
    3. CORE FUNCTIONAL REQUIREMENTS
    
    3.1 Patient Management
    - Patient registration and demographics
    - Medical history maintenance
    - Appointment scheduling
    - Patient portal for self-service
    
    3.2 Electronic Health Records (EHR)
    - Comprehensive medical records
    - Treatment history tracking
    - Medication management
    - Allergy and condition tracking
    - Document and image storage
    
    3.3 Clinical Operations
    - Doctor-patient consultation management
    - Prescription management
    - Laboratory test ordering and results
    - Radiology integration
    - Treatment plan management
    
    3.4 Billing and Insurance
    - Insurance claim processing
    - Patient billing and invoicing
    - Payment tracking and collection
    - Insurance verification
    
    3.5 Administrative Functions
    - Staff management and scheduling
    - Resource allocation
    - Inventory management for medical supplies
    - Reporting and analytics
    
    4. QUALITY AND PERFORMANCE REQUIREMENTS
    
    4.1 Performance Standards
    - System response time < 1 second for critical operations
    - 99.99% uptime for patient care systems
    - Support 5,000 concurrent users
    - Real-time data synchronization across modules
    
    4.2 Security and Compliance
    - HIPAA compliance for patient data protection
    - Role-based access control with audit trails
    - End-to-end encryption for sensitive data
    - Regular security assessments
    - Data breach prevention and response
    
    4.3 Reliability Requirements
    - 24/7 system availability
    - Automated failover capabilities
    - Data backup and disaster recovery
    - System monitoring and alerting
    
    4.4 Usability Standards
    - Intuitive interface for medical staff
    - Mobile access for doctors and nurses
    - Voice recognition for data entry
    - Customizable dashboards and workflows
    
    5. SYSTEM INTEGRATION INTERFACES
    
    5.1 Medical Device Integration
    - Laboratory equipment interfaces (HL7)
    - Radiology systems (DICOM)
    - Patient monitoring devices
    - Pharmacy management systems
    
    5.2 External System Integration
    - Insurance provider systems
    - Government health databases
    - Third-party medical applications
    - Telemedicine platforms
    
    5.3 Communication Systems
    - Email and SMS notifications
    - Alert systems for critical conditions
    - Integration with communication platforms
    
    6. DATA MANAGEMENT REQUIREMENTS
    
    6.1 Data Storage and Structure
    - Patient medical records with version control
    - Image and document management (PACS)
    - Structured and unstructured medical data
    - Historical data archiving
    
    6.2 Data Security and Privacy
    - Patient data anonymization capabilities
    - Consent management for data usage
    - Data access logging and monitoring
    - Secure data sharing protocols
    
    6.3 Data Migration and Interoperability
    - Migration from legacy systems
    - HL7 FHIR standard compliance
    - Data import/export capabilities
    - Interoperability with other healthcare systems
    
    7. PERFORMANCE AND SCALABILITY
    
    7.1 Transaction Volume
    - Handle 50,000 patient records
    - Process 1,000 appointments per day
    - Support 10,000 concurrent database transactions
    
    7.2 Storage Requirements
    - 10TB initial storage capacity
    - Scalable storage for medical images
    - Long-term archival storage (7+ years)
    
    7.3 Network and Infrastructure
    - High-availability network architecture
    - Load balancing for critical services
    - CDN for medical image delivery
    - Mobile and remote access capabilities
    
    8. REGULATORY AND COMPLIANCE
    - FDA regulations for medical software
    - HIPAA Privacy and Security Rules
    - Joint Commission standards
    - State and local healthcare regulations
    - International standards (ISO 27001)
    """
    
    # Create temporary files
    temp_files = []
    
    # Create spec file 1
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f1:
        f1.write(spec1_content)
        temp_files.append(f1.name)
    
    # Create spec file 2
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f2:
        f2.write(spec2_content)
        temp_files.append(f2.name)
    
    print(f"Created sample specification documents:")
    for i, file_path in enumerate(temp_files, 1):
        print(f"  {i}. {file_path}")
    
    return temp_files


def test_srs_generation():
    """Test the SRS generation agent with sample documents"""
    
    print("="*60)
    print("SRS Generation Agent - Testing")
    print("="*60)
    
    # Create sample specification documents
    print("\n1. Creating sample specification documents...")
    spec_files = create_sample_spec_documents()
    
    # Initialize the SRS generation agent
    print("\n2. Initializing SRS Generation Agent...")
    try:
        agent = SRSGenerationAgent(model_name="gpt-4o-mini", temperature=0.1)
        print("   Agent initialized successfully!")
    except Exception as e:
        print(f"   Error initializing agent: {e}")
        return False
    
    # Generate SRS document
    print("\n3. Generating SRS document...")
    try:
        result = agent.generate_srs(spec_files, thread_id="test_session_001")
        
        if result["success"]:
            print("   SRS generation completed successfully!")
            
            # Print summary statistics
            print(f"\n4. Generation Results Summary:")
            print(f"   - Functional Requirements: {len(result['functional_requirements'])}")
            print(f"   - Non-Functional Requirements: {len(result['non_functional_requirements'])}")
            print(f"   - System Interfaces: {len(result['system_interfaces'])}")
            print(f"   - Data Requirements: {len(result['data_requirements'])}")
            print(f"   - Performance Requirements: {len(result['performance_requirements'])}")
            print(f"   - Errors: {len(result['errors'])}")
            
            # Save the generated SRS document
            output_path = "/home/app/01_TestApp/05_Automated_AIAgentTest/generated_srs_sample.md"
            success = agent.save_srs_document(result["srs_document"], output_path)
            
            if success:
                print(f"\n5. SRS document saved to: {output_path}")
                
                # Display sample requirements
                print(f"\n6. Sample Generated Requirements:")
                
                if result['functional_requirements']:
                    print(f"\n   Functional Requirements (showing first 3):")
                    for i, req in enumerate(result['functional_requirements'][:3], 1):
                        print(f"   {i}. {req}")
                
                if result['non_functional_requirements']:
                    print(f"\n   Non-Functional Requirements (showing first 3):")
                    for i, req in enumerate(result['non_functional_requirements'][:3], 1):
                        print(f"   {i}. {req}")
                
                print(f"\n7. Requirements Analysis Summary:")
                analysis = result.get('requirements_analysis', {})
                if analysis:
                    print(f"   - Project Scope: {analysis.get('project_scope', 'N/A')[:100]}...")
                    print(f"   - Stakeholders: {len(analysis.get('stakeholders', []))}")
                    print(f"   - Objectives: {len(analysis.get('objectives', []))}")
                    print(f"   - Constraints: {len(analysis.get('constraints', []))}")
            
            # Show any errors encountered
            if result['errors']:
                print(f"\n8. Errors encountered during generation:")
                for error in result['errors']:
                    print(f"   - {error}")
            
            return True
            
        else:
            print(f"   SRS generation failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"   Error during SRS generation: {e}")
        return False
    
    finally:
        # Clean up temporary files
        print(f"\n9. Cleaning up temporary files...")
        for temp_file in spec_files:
            try:
                os.unlink(temp_file)
                print(f"   Cleaned up: {temp_file}")
            except Exception as e:
                print(f"   Error cleaning up {temp_file}: {e}")


def demonstrate_agent_capabilities():
    """Demonstrate various capabilities of the SRS generation agent"""
    
    print("="*60)
    print("SRS Generation Agent - Capability Demonstration")
    print("="*60)
    
    # Show agent architecture
    print(f"\n1. Agent Architecture Overview:")
    print(f"   - LangGraph workflow with 11 processing nodes")
    print(f"   - RAG-based specification analysis")
    print(f"   - Multi-step requirements extraction")
    print(f"   - Structured SRS document generation")
    print(f"   - Memory-based conversation state management")
    
    # Show supported document types
    print(f"\n2. Supported Input Document Types:")
    print(f"   - Plain text files (.txt)")
    print(f"   - PDF documents (.pdf)")
    print(f"   - Multiple documents per generation session")
    print(f"   - Automatic document loading and processing")
    
    # Show workflow steps
    print(f"\n3. SRS Generation Workflow Steps:")
    workflow_steps = [
        "Document Loading and Validation",
        "Text Processing and Chunking",
        "Vector Store Creation for RAG",
        "High-level Requirements Analysis",
        "Functional Requirements Extraction",
        "Non-Functional Requirements Extraction",
        "System Interface Requirements Extraction",
        "Data Requirements Extraction",
        "Performance Requirements Extraction",
        "SRS Sections Generation",
        "Final Document Compilation"
    ]
    
    for i, step in enumerate(workflow_steps, 1):
        print(f"   {i:2d}. {step}")
    
    # Show output format
    print(f"\n4. Generated SRS Document Sections:")
    srs_sections = [
        "1. Introduction (Purpose, Scope, Definitions, References)",
        "2. Overall Description (Product Perspective, Functions, Users)",
        "3. Functional Requirements (Feature specifications)",
        "4. Non-Functional Requirements (Quality attributes)",
        "5. System Interfaces (External integrations)",
        "6. Data Requirements (Storage and management)",
        "7. Performance Requirements (Metrics and targets)"
    ]
    
    for section in srs_sections:
        print(f"   - {section}")
    
    # Show configuration options
    print(f"\n5. Configuration Options:")
    print(f"   - Model Selection: GPT-4o, GPT-4o-mini, or other OpenAI models")
    print(f"   - Temperature Control: Adjustable creativity level")
    print(f"   - Chunk Size: Configurable document processing")
    print(f"   - Retrieval Parameters: MMR search with configurable k values")
    print(f"   - Memory Management: Session-based state persistence")
    
    print(f"\n6. Use Cases and Applications:")
    use_cases = [
        "Software development project specifications",
        "System integration requirements documentation",
        "Enterprise software requirements analysis",
        "Regulatory compliance documentation",
        "Technical specification standardization",
        "Requirements traceability and management"
    ]
    
    for i, use_case in enumerate(use_cases, 1):
        print(f"   {i}. {use_case}")


def test_claude_integration():
    """Test Claude model integration"""
    print("\n🧪 Testing Claude Model Integration")
    print("-" * 50)
    
    try:
        # Test with Claude model
        config = AgentConfig(ConfigProfiles.claude_production())
        agent = SRSGenerationAgent(
            model_name=config.model.name,
            temperature=config.model.temperature
        )
        
        print(f"✅ Successfully initialized agent with Claude model: {config.model.name}")
        
        # Create sample document
        sample_files = create_sample_spec_documents()
        
        # Test SRS generation with Claude
        print("🔄 Generating SRS with Claude model...")
        result = agent.generate_srs(sample_files[:1])  # Use only first sample to save tokens
        
        if result and result.get("final_srs"):
            print("✅ Claude model integration successful!")
            print(f"Generated SRS length: {len(result['final_srs'])} characters")
            return True
        else:
            print("❌ Claude model integration failed - no SRS generated")
            return False
            
    except Exception as e:
        print(f"❌ Claude model integration failed: {str(e)}")
        return False
    finally:
        # Clean up sample files
        for file_path in sample_files:
            try:
                os.unlink(file_path)
            except:
                pass


if __name__ == "__main__":
    import sys
    
    # Check API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if not has_openai and not has_anthropic:
        print("Warning: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY environment variable is set.")
        print("Please set at least one API key to test the agent.")
        print("Examples:")
        print("  export OPENAI_API_KEY='your-openai-api-key-here'")
        print("  export ANTHROPIC_API_KEY='your-anthropic-api-key-here'")
        sys.exit(1)
    
    print("🚀 SRS Generation Agent - Comprehensive Testing")
    print("=" * 60)
    
    if has_openai:
        print("\n📋 Testing with OpenAI models...")
        # Run demonstration
        demonstrate_agent_capabilities()
        
        # Run test
        print("\n" + "="*60)
        test_result = test_srs_generation()
    
    if has_anthropic:
        print("\n🤖 Testing with Claude models...")
        claude_result = test_claude_integration()
    else:
        claude_result = None
    
    # Final results
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("-" * 30)
    
    if has_openai:
        status = "✅ PASSED" if test_result else "❌ FAILED"
        print(f"OpenAI Models: {status}")
    
    if has_anthropic:
        status = "✅ PASSED" if claude_result else "❌ FAILED"
        print(f"Claude Models: {status}")
    
    overall_success = (not has_openai or test_result) and (not has_anthropic or claude_result)
    
    if overall_success:
        print("\n🎉 All tests completed successfully!")
        print("The SRS Generation Agent is ready for production use.")
    else:
        print("\n⚠️  Some tests failed.")
        print("Please check the error messages above for troubleshooting.")