"""Factory for creating iteration 1 chat implementations."""

from enum import Enum

from account_intelligence_ai.core.chat_interface import ChatInterface
from account_intelligence_ai.iteration1.part1 import WebSearchChat
from account_intelligence_ai.iteration1.part2 import DocumentRAGChat
from account_intelligence_ai.iteration1.part3 import CorrectiveRAGChat

class Iteration1Mode(Enum):
    """Modes corresponding to the three parts of iteration 1 (seach / RAG / Corrective RAG (search+RAG))."""
    PART1_WEB_SEARCH = "part1"      # Web search using Tavily
    PART2_DOCUMENT_RAG = "part2"    # RAG with OPM documents
    PART3_CORRECTIVE_RAG = "part3"  # Corrective RAG combining both approaches

def create_chat_implementation(mode: Iteration1Mode) -> ChatInterface:
    """Create and return the appropriate chat implementation.
    
    Args:
        mode: Which part of iteration 1 to run
        
    Returns:
        ChatInterface: The appropriate chat implementation
    
    Raises:
        ValueError: If mode is not recognized
    """
    implementations = {
        Iteration1Mode.PART1_WEB_SEARCH: WebSearchChat,
        Iteration1Mode.PART2_DOCUMENT_RAG: DocumentRAGChat,
        Iteration1Mode.PART3_CORRECTIVE_RAG: CorrectiveRAGChat
    }
    
    if mode not in implementations:
        raise ValueError(f"Unknown mode: {mode}")
    
    implementation_class = implementations[mode]
    implementation = implementation_class()
    return implementation 