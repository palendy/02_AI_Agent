"""
Comprehensive logging and error handling system for SRS Generation Agent.
Provides structured logging, error tracking, and monitoring capabilities.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import structlog
from rich.logging import RichHandler
from rich.console import Console

from config import get_config

# Initialize console for rich output
console = Console()


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """Component type enumeration for structured logging."""
    AGENT = "agent"
    PARSER = "parser"
    VALIDATOR = "validator"
    TEMPLATE = "template"
    CLI = "cli"
    CONFIG = "config"
    CORE = "core"


@dataclass
class LogContext:
    """Context information for structured logging."""
    component: ComponentType
    operation: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    project_name: Optional[str] = None
    file_path: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_type: str
    error_message: str
    traceback_str: str
    component: ComponentType
    operation: str
    timestamp: str
    context: Dict[str, Any]
    severity: str


class SRSLogger:
    """Main logger class for SRS Generation Agent."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = None
        self.error_count = 0
        self.warning_count = 0
        self.session_id = self._generate_session_id()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging configuration."""
        try:
            # Configure structlog
            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.TimeStamper(fmt="ISO"),
                    structlog.processors.add_log_level,
                    structlog.processors.StackInfoRenderer(),
                    structlog.dev.ConsoleRenderer() if self.config.logging.debug else structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            
            # Setup standard logging
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, self.config.logging.log_level))
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Console handler with rich formatting
            console_handler = RichHandler(
                console=console,
                show_time=True,
                show_level=True,
                show_path=self.config.logging.debug,
                rich_tracebacks=True,
                markup=True
            )
            console_handler.setLevel(getattr(logging, self.config.logging.log_level))
            console_handler.setFormatter(logging.Formatter(
                '%(message)s'
            ))
            root_logger.addHandler(console_handler)
            
            # File handler if configured
            if self.config.logging.log_file:
                file_handler = self._setup_file_handler()
                root_logger.addHandler(file_handler)
            
            # Create structured logger
            self.logger = structlog.get_logger("srs_agent")
            
            # Log initialization
            self.logger.info(
                "Logger initialized",
                session_id=self.session_id,
                log_level=self.config.logging.log_level,
                file_logging=bool(self.config.logging.log_file),
                debug_mode=self.config.logging.debug
            )
            
        except Exception as e:
            # Fallback to basic logging if setup fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.error(f"Failed to setup advanced logging: {str(e)}")
            self.logger = logging.getLogger("srs_agent")
    
    def _setup_file_handler(self) -> logging.Handler:
        """Setup file logging handler."""
        log_file = Path(self.config.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler for production
        try:
            from logging.handlers import RotatingFileHandler
            handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
        except ImportError:
            # Fallback to regular file handler
            handler = logging.FileHandler(log_file)
        
        handler.setLevel(logging.DEBUG)  # File gets all messages
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        
        return handler
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"srs_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def set_context(self, context: LogContext):
        """Set logging context for structured logging."""
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            component=context.component.value,
            operation=context.operation,
            session_id=context.session_id or self.session_id,
            user_id=context.user_id,
            project_name=context.project_name,
            file_path=context.file_path,
            request_id=context.request_id
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.warning_count += 1
        self.logger.warning(message, warning_count=self.warning_count, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details."""
        self.error_count += 1
        
        error_info = {}
        if error:
            error_info.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc() if self.config.logging.debug else None
            })
        
        self.logger.error(
            message,
            error_count=self.error_count,
            **error_info,
            **kwargs
        )
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        error_info = {}
        if error:
            error_info.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            })
        
        self.logger.critical(message, **error_info, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_user_action(self, action: str, details: Dict[str, Any] = None):
        """Log user actions for audit trail."""
        self.logger.info(
            "User action",
            action=action,
            details=details or {},
            audit=True
        )
    
    def log_agent_decision(self, decision: str, reasoning: str, confidence: float):
        """Log AI agent decisions for transparency."""
        self.logger.info(
            "Agent decision",
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            agent_log=True
        )
    
    def log_validation_result(self, component: str, result: Dict[str, Any]):
        """Log validation results."""
        self.logger.info(
            "Validation result",
            component=component,
            score=result.get("score"),
            passed=result.get("passed"),
            issues=len(result.get("issues", [])),
            validation=True
        )
    
    def capture_exception(self, error: Exception, context: Optional[LogContext] = None) -> ErrorInfo:
        """Capture detailed exception information."""
        if context:
            self.set_context(context)
        
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            component=context.component if context else ComponentType.CORE,
            operation=context.operation if context else "unknown",
            timestamp=datetime.now().isoformat(),
            context=dict(structlog.contextvars.get_contextvars()),
            severity="critical" if isinstance(error, (SystemError, MemoryError)) else "error"
        )
        
        self.error(
            f"Exception captured: {error_info.error_message}",
            error=error,
            error_info=error_info.__dict__
        )
        
        return error_info
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get logging statistics for current session."""
        return {
            "session_id": self.session_id,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "log_level": self.config.logging.log_level,
            "session_start": datetime.now().isoformat()
        }


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger()
    
    @property
    def logger(self) -> SRSLogger:
        """Get logger instance."""
        return self._logger
    
    def log_method_entry(self, method_name: str, **kwargs):
        """Log method entry."""
        self._logger.debug(f"Entering {method_name}", method=method_name, **kwargs)
    
    def log_method_exit(self, method_name: str, result: Any = None, **kwargs):
        """Log method exit."""
        self._logger.debug(
            f"Exiting {method_name}",
            method=method_name,
            result_type=type(result).__name__ if result is not None else None,
            **kwargs
        )
    
    def log_method_error(self, method_name: str, error: Exception, **kwargs):
        """Log method error."""
        self._logger.error(
            f"Error in {method_name}",
            method=method_name,
            error=error,
            **kwargs
        )


# Context managers for logging
@contextmanager
def log_operation(operation: str, component: ComponentType, logger: SRSLogger = None, **context_kwargs):
    """Context manager for logging operations with timing."""
    if logger is None:
        logger = get_logger()
    
    context = LogContext(
        component=component,
        operation=operation,
        **context_kwargs
    )
    logger.set_context(context)
    
    start_time = datetime.now()
    logger.info(f"Starting {operation}")
    
    try:
        yield logger
        duration = (datetime.now() - start_time).total_seconds()
        logger.log_performance(operation, duration)
        logger.info(f"Completed {operation}", duration_seconds=duration)
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_info = logger.capture_exception(e, context)
        logger.critical(
            f"Failed {operation}",
            duration_seconds=duration,
            error_info=error_info.__dict__
        )
        raise


@contextmanager
def suppress_and_log(error_types: tuple = (Exception,), 
                    default_return=None, 
                    logger: SRSLogger = None,
                    component: ComponentType = ComponentType.CORE):
    """Context manager to suppress specific errors and log them."""
    if logger is None:
        logger = get_logger()
    
    try:
        yield
    except error_types as e:
        context = LogContext(component=component, operation="error_suppression")
        logger.capture_exception(e, context)
        return default_return


# Performance monitoring decorator
def monitor_performance(component: ComponentType = ComponentType.CORE):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            with log_operation(func.__name__, component, logger):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_exceptions(component: ComponentType = ComponentType.CORE, 
                  reraise: bool = True):
    """Decorator to automatically log exceptions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = LogContext(
                    component=component,
                    operation=func.__name__
                )
                logger.capture_exception(e, context)
                
                if reraise:
                    raise
                return None
        
        return wrapper
    return decorator


# Global logger instance
_global_logger = None


def get_logger() -> SRSLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = SRSLogger()
    return _global_logger


def setup_logging(config=None) -> SRSLogger:
    """Setup and return logger instance."""
    global _global_logger
    _global_logger = SRSLogger(config)
    return _global_logger


def reset_logger():
    """Reset global logger instance."""
    global _global_logger
    _global_logger = None


# Error handling utilities
class SRSError(Exception):
    """Base exception class for SRS Agent errors."""
    
    def __init__(self, message: str, component: ComponentType, details: Dict[str, Any] = None):
        self.message = message
        self.component = component
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)


class ConfigurationError(SRSError):
    """Configuration-related errors."""
    pass


class DocumentParsingError(SRSError):
    """Document parsing errors."""
    pass


class ValidationError(SRSError):
    """Validation-related errors."""
    pass


class GenerationError(SRSError):
    """SRS generation errors."""
    pass


class APIError(SRSError):
    """API-related errors."""
    pass


def handle_error(error: Exception, 
                component: ComponentType = ComponentType.CORE,
                operation: str = "unknown",
                logger: SRSLogger = None) -> ErrorInfo:
    """Centralized error handling function."""
    if logger is None:
        logger = get_logger()
    
    context = LogContext(component=component, operation=operation)
    return logger.capture_exception(error, context)


# Graceful shutdown handling
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    import signal
    
    def signal_handler(signum, frame):
        logger = get_logger()
        logger.critical(f"Received signal {signum}, shutting down gracefully")
        
        # Log final statistics
        stats = logger.get_session_stats()
        logger.info("Session statistics", **stats)
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Health check functionality
def health_check() -> Dict[str, Any]:
    """Perform system health check."""
    logger = get_logger()
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "logger_status": "healthy",
        "session_stats": logger.get_session_stats(),
        "system": {
            "python_version": sys.version,
            "platform": sys.platform,
            "memory_usage": _get_memory_usage()
        }
    }
    
    logger.info("Health check completed", **health_status)
    return health_status


def _get_memory_usage() -> Dict[str, Union[int, str]]:
    """Get current memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {"status": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


# Export main functions and classes
__all__ = [
    'SRSLogger', 'LogContext', 'ErrorInfo', 'ComponentType', 'LogLevel',
    'LoggerMixin', 'log_operation', 'suppress_and_log', 'monitor_performance',
    'log_exceptions', 'get_logger', 'setup_logging', 'reset_logger',
    'SRSError', 'ConfigurationError', 'DocumentParsingError', 'ValidationError',
    'GenerationError', 'APIError', 'handle_error', 'setup_signal_handlers',
    'health_check'
]