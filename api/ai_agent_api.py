"""
API endpoints for AI SQL Agent
FastAPI endpoints for chat interface with AI Agent
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from services.ai_sql_agent import AISQLAgent

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/ai-agent", tags=["AI Agent"])


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    question: str = Field(..., min_length=1, max_length=1000, description="User question in Vietnamese or English")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    user_question: str
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    results_count: Optional[int] = None
    insights: Optional[str] = None
    recommendations: Optional[List[str]] = None
    ai_response: Optional[str] = None
    execution_time_ms: Optional[int] = None
    session_id: Optional[str] = None
    error: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Response model for chat history"""
    session_id: str
    messages: List[Dict[str, Any]]
    total_count: int


class SuggestedQuestionsResponse(BaseModel):
    """Response model for suggested questions"""
    questions: List[str]


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    agent_ready: bool


# Dependency to get AI Agent instance
def get_ai_agent() -> AISQLAgent:
    """Dependency to create AI Agent instance"""
    try:
        return AISQLAgent()
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="AI Agent initialization failed. Please check OpenAI API key configuration."
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: AISQLAgent = Depends(get_ai_agent)
) -> ChatResponse:
    """
    Process natural language question and return AI-generated insights

    Args:
        request: ChatRequest with user question and optional session_id

    Returns:
        ChatResponse with SQL query, results, insights, and recommendations
    """
    try:
        result = agent.process_query(
            user_question=request.question,
            session_id=request.session_id
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(
    session_id: str,
    limit: int = 20,
    agent: AISQLAgent = Depends(get_ai_agent)
) -> ChatHistoryResponse:
    """
    Get chat history for a specific session

    Args:
        session_id: Session ID to retrieve history for
        limit: Maximum number of messages to return (default 20)

    Returns:
        ChatHistoryResponse with conversation history
    """
    try:
        messages = agent.get_chat_history(session_id, limit)

        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_count=len(messages)
        )

    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.get("/suggested-questions", response_model=SuggestedQuestionsResponse)
async def get_suggested_questions(
    agent: AISQLAgent = Depends(get_ai_agent)
) -> SuggestedQuestionsResponse:
    """
    Get list of suggested questions for users

    Returns:
        SuggestedQuestionsResponse with list of example questions
    """
    try:
        questions = agent.get_suggested_questions()

        return SuggestedQuestionsResponse(questions=questions)

    except Exception as e:
        logger.error(f"Error getting suggested questions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving questions: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint for AI Agent service

    Returns:
        HealthCheckResponse with service status
    """
    try:
        # Try to initialize agent to check if everything is configured
        agent = AISQLAgent()
        agent_ready = True
    except Exception as e:
        logger.warning(f"AI Agent not ready: {str(e)}")
        agent_ready = False

    return HealthCheckResponse(
        status="healthy" if agent_ready else "degraded",
        timestamp=datetime.now().isoformat(),
        agent_ready=agent_ready
    )


# Additional utility endpoints

@router.post("/execute-sql")
async def execute_custom_sql(
    sql_query: str,
    agent: AISQLAgent = Depends(get_ai_agent)
) -> Dict[str, Any]:
    """
    Execute custom SQL query (for testing/debugging only)
    Should be protected in production!

    Args:
        sql_query: SQL query to execute

    Returns:
        Query results
    """
    try:
        # Validate query is safe
        if not agent._is_safe_query(sql_query):
            raise HTTPException(
                status_code=400,
                detail="Only SELECT queries are allowed"
            )

        results = agent.db.fetch_all(sql_query)

        return {
            'success': True,
            'query': sql_query,
            'results': results,
            'count': len(results)
        }

    except Exception as e:
        logger.error(f"Error executing custom SQL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing query: {str(e)}"
        )


@router.get("/stats")
async def get_agent_stats(
    agent: AISQLAgent = Depends(get_ai_agent)
) -> Dict[str, Any]:
    """
    Get statistics about AI Agent usage

    Returns:
        Statistics dictionary
    """
    try:
        # Get total chat interactions
        total_chats_query = "SELECT COUNT(*) as total FROM ai_chat_history"
        total_chats = agent.db.fetch_one(total_chats_query)

        # Get successful queries
        success_query = "SELECT COUNT(*) as successful FROM ai_chat_history WHERE is_successful = TRUE"
        successful = agent.db.fetch_one(success_query)

        # Get average execution time
        avg_time_query = "SELECT AVG(execution_time_ms) as avg_time FROM ai_chat_history WHERE is_successful = TRUE"
        avg_time = agent.db.fetch_one(avg_time_query)

        # Get recent sessions
        recent_sessions_query = """
            SELECT COUNT(DISTINCT session_id) as sessions
            FROM ai_chat_history
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """
        recent_sessions = agent.db.fetch_one(recent_sessions_query)

        return {
            'total_interactions': total_chats.get('total', 0) if total_chats else 0,
            'successful_queries': successful.get('successful', 0) if successful else 0,
            'average_execution_time_ms': round(avg_time.get('avg_time', 0), 2) if avg_time else 0,
            'recent_sessions_24h': recent_sessions.get('sessions', 0) if recent_sessions else 0,
            'success_rate': round(
                (successful.get('successful', 0) / total_chats.get('total', 1) * 100)
                if total_chats and total_chats.get('total', 0) > 0 else 0,
                2
            )
        }

    except Exception as e:
        logger.error(f"Error getting agent stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )
