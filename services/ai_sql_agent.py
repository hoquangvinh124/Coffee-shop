"""
AI SQL Agent - Smart query generator and analyzer for coffee shop predictions
Uses OpenAI GPT to convert natural language to SQL and provide insights
"""
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from openai import OpenAI
from utils.database import DatabaseManager
from utils.config import OPENAI_API_KEY, OPENAI_MODEL, AI_AGENT_TEMPERATURE, AI_AGENT_MAX_TOKENS

logger = logging.getLogger(__name__)


class AISQLAgent:
    """
    AI Agent that can:
    1. Convert natural language questions to SQL queries
    2. Execute queries safely
    3. Analyze results and provide insights in Vietnamese
    4. Give recommendations for coffee shop operations
    """

    def __init__(self, api_key: str = None):
        """Initialize AI SQL Agent with OpenAI client"""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in config or .env")

        self.client = OpenAI(api_key=self.api_key)
        self.db = DatabaseManager()
        self.model = OPENAI_MODEL

        # Try to load ML predictor for live predictions
        try:
            from revenue_forecasting.predictor import RevenuePredictor
            self.predictor = RevenuePredictor()
            self.can_predict = True
            logger.info("ML Predictor loaded - can generate live predictions")
        except Exception as e:
            self.predictor = None
            self.can_predict = False
            logger.warning(f"ML Predictor not available: {e}")

        # Database schema information for AI context
        self.schema_context = self._build_schema_context()

    def _build_schema_context(self) -> str:
        """Build database schema context for AI"""
        return """
Database Schema for Coffee Shop Revenue Forecasting System:

1. STORE_METADATA table (store_metadata):
   - store_nbr: INT (Primary Key) - Store number
   - city: VARCHAR - City where store is located
   - state: VARCHAR - State/Province
   - type: VARCHAR - Store type (A, B, C, D, E)
   - cluster: INT - Store cluster for grouping
   - total_revenue: DECIMAL - Total historical revenue ($)
   - avg_daily_sales: DECIMAL - Average daily sales ($)
   - std_sales: DECIMAL - Standard deviation of sales
   - total_transactions: BIGINT - Total number of transactions

2. OVERALL_PREDICTIONS table (overall_predictions):
   - id: INT (Primary Key)
   - ds: DATE - Prediction date (Prophet standard column)
   - yhat: DECIMAL - Predicted revenue ($)
   - yhat_lower: DECIMAL - Lower confidence bound ($)
   - yhat_upper: DECIMAL - Upper confidence bound ($)
   - trend: DECIMAL - Trend component
   - weekly: DECIMAL - Weekly seasonality
   - yearly: DECIMAL - Yearly seasonality
   - is_historical: BOOLEAN - TRUE if past data, FALSE if future forecast
   - created_at: TIMESTAMP

3. STORE_PREDICTIONS table (store_predictions):
   - id: INT (Primary Key)
   - store_nbr: INT (Foreign Key to store_metadata)
   - ds: DATE - Prediction date
   - yhat: DECIMAL - Predicted revenue for this store ($)
   - yhat_lower: DECIMAL - Lower bound
   - yhat_upper: DECIMAL - Upper bound
   - is_historical: BOOLEAN - Historical vs future
   - created_at: TIMESTAMP

4. YEARLY_FORECAST_SUMMARY table (yearly_forecast_summary):
   - year: INT (Primary Key)
   - avg_daily: DECIMAL - Average daily revenue for year
   - total: DECIMAL - Total revenue for year
   - std: DECIMAL - Standard deviation
   - total_lower: DECIMAL - Lower bound
   - total_upper: DECIMAL - Upper bound
   - total_m: DECIMAL - Total in millions

5. MODEL_METRICS table (model_metrics):
   - id: INT (Primary Key)
   - model_type: VARCHAR - 'overall' or 'store'
   - store_nbr: INT - NULL for overall model
   - metric_name: VARCHAR - 'MAE', 'MAPE', 'RMSE', 'Coverage'
   - metric_value: DECIMAL
   - unit: VARCHAR - '$' or '%'

6. AI_INSIGHTS table (ai_insights):
   - id: INT (Primary Key)
   - insight_type: VARCHAR - 'revenue_alert', 'trend_change', 'recommendation'
   - store_nbr: INT - Related store (NULL for system-wide)
   - title: VARCHAR - Insight title
   - description: TEXT - Detailed description
   - severity: VARCHAR - 'info', 'warning', 'critical'
   - metadata: JSON - Additional data
   - is_read: BOOLEAN
   - created_at: TIMESTAMP

USEFUL VIEWS:
- v_next_7days_forecast: Forecast for next 7 days
- v_store_performance: Store ranking by revenue
- v_monthly_forecast: Monthly revenue aggregates
- v_unread_insights: Unread insights/alerts

IMPORTANT NOTES:
- Revenue is in USD ($), not VND
- Use 'ds' for date columns (Prophet standard)
- Use 'yhat' for predicted values (Prophet standard)
- CURDATE() returns today's date
- Filter is_historical=FALSE for future predictions
- Filter is_historical=TRUE for historical data
- Always use DATE format: 'YYYY-MM-DD'
- DAYNAME(ds) returns day name (Monday, Tuesday, etc.)
- DAYOFWEEK(ds) returns day number (1=Sunday, 7=Saturday)
"""

    def _create_system_prompt(self) -> str:
        """Create system prompt for AI agent"""
        return f"""Bạn là một AI Assistant chuyên nghiệp cho hệ thống quản lý chuỗi quán cà phê.

NHIỆM VỤ:
1. Chuyển đổi câu hỏi tiếng Việt/tiếng Anh thành SQL query chính xác
2. Phân tích kết quả và đưa ra insights hữu ích
3. Đề xuất hành động cụ thể cho quán cà phê

{self.schema_context}

QUY TẮC QUAN TRỌNG:
1. CHỈ tạo SELECT queries - KHÔNG BAO GIỜ tạo INSERT, UPDATE, DELETE, DROP
2. LUÔN giới hạn kết quả với LIMIT (mặc định 100)
3. Sử dụng JOIN khi cần dữ liệu từ nhiều bảng
4. Format số tiền: thêm dấu phẩy và đơn vị VNĐ
5. Trả lời HOÀN TOÀN bằng tiếng Việt, chuyên nghiệp và dễ hiểu
6. Đưa ra recommendations cụ thể và khả thi

FORMAT TRẢ LỜI:
Trả về JSON với format sau:
{{
    "sql_query": "SELECT ... (SQL query chuẩn)",
    "explanation": "Giải thích ngắn gọn query này làm gì",
    "insights": "Phân tích insights từ dữ liệu (sau khi có kết quả)",
    "recommendations": ["Gợi ý 1", "Gợi ý 2", "..."]
}}

VÍ DỤ:
User: "Doanh thu dự đoán cho tuần tới là bao nhiêu?"
AI: {{
    "sql_query": "SELECT prediction_date, SUM(predicted_revenue) as total_revenue FROM predictions WHERE prediction_date BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 7 DAY) GROUP BY prediction_date ORDER BY prediction_date LIMIT 7",
    "explanation": "Query này tính tổng doanh thu dự đoán cho 7 ngày tới",
    "insights": "Sẽ phân tích sau khi có dữ liệu",
    "recommendations": []
}}
"""

    def process_query(self, user_question: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main method to process user question

        Args:
            user_question: Natural language question in Vietnamese or English
            session_id: Optional session ID for tracking conversation

        Returns:
            Dictionary with query results, insights, and recommendations
        """
        start_time = time.time()
        session_id = session_id or f"session_{int(time.time())}"

        try:
            # Step 0: Check data availability
            data_check = self.check_data_availability()

            if not data_check.get('has_data', False):
                # Auto-generate predictions if none exist
                logger.info("No prediction data found. Attempting auto-generation...")

                # Try to auto-generate using ML models
                if self.can_predict and self.predictor:
                    try:
                        logger.info("Auto-generating predictions using Prophet models...")

                        # Import auto generator
                        from services.auto_prediction_generator import AutoPredictionGenerator

                        generator = AutoPredictionGenerator()

                        # Generate and import (limited to 10 stores for speed)
                        stats = generator.auto_generate_and_import(
                            days_future=90,  # 3 months
                            max_stores=10    # Limit to 10 stores for speed
                        )

                        if stats.get('success'):
                            logger.info("Auto-generation successful! Proceeding with query...")
                            # Re-check data availability
                            data_check = self.check_data_availability()

                            if not data_check.get('has_data', False):
                                raise Exception("Data generation succeeded but still no data available")

                            # Continue to process the query (don't return here)
                        else:
                            raise Exception("Auto-generation failed")

                    except Exception as e:
                        logger.error(f"Auto-generation failed: {str(e)}")
                        error_msg = (
                            f"⚠️ Không tìm thấy dữ liệu và không thể tự động generate!\n\n"
                            f"Lỗi: {str(e)}\n\n"
                            f"Vui lòng chạy thủ công:\n"
                            f"  python database/import_predictions_to_db.py\n\n"
                            f"Hoặc:\n"
                            f"  python services/auto_prediction_generator.py"
                        )

                        return {
                            'success': False,
                            'error': error_msg,
                            'user_question': user_question,
                            'data_check': data_check
                        }
                else:
                    # ML predictor not available
                    error_msg = (
                        "⚠️ Không tìm thấy dữ liệu dự đoán trong database!\n\n"
                        "Vui lòng chạy lệnh sau:\n\n"
                        "1. Import từ CSV (nhanh):\n"
                        "   python database/import_predictions_to_db.py\n\n"
                        "2. Tự động generate từ models:\n"
                        "   python services/auto_prediction_generator.py\n\n"
                        f"Hiện tại: {data_check.get('overall_predictions', 0)} predictions, "
                        f"{data_check.get('stores_count', 0)} stores"
                    )

                    return {
                        'success': False,
                        'error': error_msg,
                        'user_question': user_question,
                        'data_check': data_check,
                        'can_auto_generate': True
                    }

            # Step 1: Generate SQL query using AI
            logger.info(f"Processing question: {user_question}")
            sql_response = self._generate_sql(user_question)

            if not sql_response or 'sql_query' not in sql_response:
                return {
                    'success': False,
                    'error': 'Không thể tạo SQL query từ câu hỏi này',
                    'user_question': user_question
                }

            sql_query = sql_response['sql_query'].strip()

            # Validate SQL (basic security check)
            if not self._is_safe_query(sql_query):
                return {
                    'success': False,
                    'error': 'Query không an toàn - chỉ chấp nhận SELECT queries',
                    'user_question': user_question
                }

            # Step 2: Execute SQL query
            logger.info(f"Executing SQL: {sql_query}")
            query_results = self.db.fetch_all(sql_query)

            # Step 3: Generate insights from results
            insights_response = self._generate_insights(
                user_question,
                sql_query,
                query_results,
                sql_response.get('explanation', '')
            )

            execution_time = int((time.time() - start_time) * 1000)

            # Step 4: Save to chat history
            self._save_chat_history(
                session_id=session_id,
                user_message=user_question,
                ai_response=insights_response['ai_response'],
                sql_query=sql_query,
                query_results=query_results,
                execution_time_ms=execution_time,
                is_successful=True
            )

            return {
                'success': True,
                'user_question': user_question,
                'sql_query': sql_query,
                'explanation': sql_response.get('explanation', ''),
                'results': query_results,
                'results_count': len(query_results),
                'insights': insights_response.get('insights', ''),
                'recommendations': insights_response.get('recommendations', []),
                'ai_response': insights_response['ai_response'],
                'execution_time_ms': execution_time,
                'session_id': session_id
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            execution_time = int((time.time() - start_time) * 1000)

            # Save error to history
            self._save_chat_history(
                session_id=session_id,
                user_message=user_question,
                ai_response=f"Lỗi: {str(e)}",
                sql_query=None,
                query_results=None,
                execution_time_ms=execution_time,
                is_successful=False,
                error_message=str(e)
            )

            return {
                'success': False,
                'error': str(e),
                'user_question': user_question,
                'execution_time_ms': execution_time
            }

    def _generate_sql(self, user_question: str) -> Optional[Dict]:
        """Generate SQL query from natural language using AI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": user_question}
                ],
                temperature=AI_AGENT_TEMPERATURE,
                max_tokens=AI_AGENT_MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"Generated SQL: {result.get('sql_query', 'N/A')}")
            return result

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return None

    def _generate_insights(
        self,
        user_question: str,
        sql_query: str,
        results: List[Dict],
        explanation: str
    ) -> Dict[str, Any]:
        """Generate insights and recommendations from query results"""
        try:
            # Prepare results summary for AI
            results_summary = self._prepare_results_summary(results)

            prompt = f"""
Câu hỏi của người dùng: {user_question}

SQL Query đã thực thi: {sql_query}

Kết quả ({len(results)} dòng):
{results_summary}

NHIỆM VỤ:
1. Phân tích kết quả và đưa ra insights chi tiết
2. Đề xuất hành động cụ thể cho quán cà phê dựa trên dữ liệu
3. Trả lời hoàn toàn bằng tiếng Việt, dễ hiểu

Trả về JSON format:
{{
    "ai_response": "Câu trả lời đầy đủ, chi tiết cho người dùng",
    "insights": "Phân tích insights từ dữ liệu",
    "recommendations": ["Gợi ý 1", "Gợi ý 2", "..."]
}}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu cho chuỗi quán cà phê."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=AI_AGENT_MAX_TOKENS,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                'ai_response': f"Tìm thấy {len(results)} kết quả.",
                'insights': 'Không thể tạo insights tự động.',
                'recommendations': []
            }

    def _prepare_results_summary(self, results: List[Dict], max_rows: int = 50) -> str:
        """Prepare a summary of results for AI analysis"""
        if not results:
            return "Không có dữ liệu"

        # Limit rows for AI context
        limited_results = results[:max_rows]

        # Format as readable text
        summary = []
        for i, row in enumerate(limited_results, 1):
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            summary.append(f"{i}. {row_text}")

        if len(results) > max_rows:
            summary.append(f"... và {len(results) - max_rows} dòng khác")

        return "\n".join(summary)

    def _is_safe_query(self, sql_query: str) -> bool:
        """Validate SQL query for safety"""
        sql_upper = sql_query.upper().strip()

        # Only allow SELECT queries
        if not sql_upper.startswith('SELECT'):
            return False

        # Block dangerous keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER',
            'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC',
            'EXECUTE', 'CALL', 'INTO OUTFILE', 'INTO DUMPFILE'
        ]

        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False

        return True

    def _save_chat_history(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        sql_query: Optional[str],
        query_results: Optional[List[Dict]],
        execution_time_ms: int,
        is_successful: bool,
        error_message: Optional[str] = None
    ) -> bool:
        """Save conversation to database"""
        try:
            query = """
                INSERT INTO ai_chat_history
                (session_id, user_message, ai_response, sql_query, query_results,
                 execution_time_ms, is_successful, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """

            results_json = json.dumps(query_results, ensure_ascii=False) if query_results else None

            self.db.execute_query(query, (
                session_id,
                user_message,
                ai_response,
                sql_query,
                results_json,
                execution_time_ms,
                is_successful,
                error_message
            ))

            return True

        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return False

    def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Retrieve chat history for a session"""
        query = """
            SELECT id, user_message, ai_response, sql_query,
                   execution_time_ms, is_successful, created_at
            FROM ai_chat_history
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """

        results = self.db.fetch_all(query, (session_id, limit))
        return list(reversed(results))  # Return in chronological order

    def get_suggested_questions(self) -> List[str]:
        """Get list of suggested questions for users"""
        return [
            # Overall Revenue Queries
            "Doanh thu dự đoán tuần tới là bao nhiêu?",
            "Dự đoán doanh thu tháng tới",
            "Dự đoán doanh thu cuối tuần này",
            "Tổng doanh thu dự đoán 30 ngày tới",

            # Store-Specific Queries
            "Cửa hàng nào có doanh thu cao nhất?",
            "Top 5 cửa hàng có doanh thu tốt nhất",
            "Cửa hàng nào cần cải thiện hiệu suất?",
            "Dự đoán doanh thu cửa hàng số 44 tuần tới",
            "So sánh doanh thu giữa các cửa hàng",

            # Trend Analysis
            "Phân tích xu hướng doanh thu 30 ngày qua",
            "Ngày nào trong tuần có doanh thu cao nhất?",
            "Doanh thu cuối tuần so với ngày thường như thế nào?",
            "Xu hướng doanh thu theo tháng",

            # Custom Time Range
            "Dự đoán doanh thu từ ngày 2025-01-20 đến 2025-01-31",
            "Doanh thu dự kiến quý 1 năm 2025",

            # Insights & Recommendations
            "Có insights nào quan trọng không?",
            "Những cửa hàng nào đang có xu hướng tăng trưởng?",
            "Hiệu suất dự đoán của model như thế nào?"
        ]

    def check_data_availability(self) -> Dict[str, Any]:
        """Check if there's any prediction data in the database"""
        try:
            # Check store metadata
            stores_count = self.db.fetch_one("SELECT COUNT(*) as count FROM store_metadata")

            # Check overall predictions
            overall_count = self.db.fetch_one("SELECT COUNT(*) as count FROM overall_predictions")

            # Check store predictions
            store_pred_count = self.db.fetch_one("SELECT COUNT(*) as count FROM store_predictions")

            # Check date range
            date_range = self.db.fetch_one("""
                SELECT MIN(ds) as first_date, MAX(ds) as last_date
                FROM overall_predictions
            """)

            # Check future predictions
            future_count = self.db.fetch_one("""
                SELECT COUNT(*) as count
                FROM overall_predictions
                WHERE ds >= CURDATE() AND is_historical = FALSE
            """)

            has_data = (
                stores_count and stores_count['count'] > 0 and
                overall_count and overall_count['count'] > 0
            )

            return {
                'has_data': has_data,
                'stores_count': stores_count['count'] if stores_count else 0,
                'overall_predictions': overall_count['count'] if overall_count else 0,
                'store_predictions': store_pred_count['count'] if store_pred_count else 0,
                'future_predictions': future_count['count'] if future_count else 0,
                'date_range': {
                    'first': str(date_range['first_date']) if date_range and date_range['first_date'] else None,
                    'last': str(date_range['last_date']) if date_range and date_range['last_date'] else None
                } if date_range else None
            }

        except Exception as e:
            logger.error(f"Error checking data availability: {str(e)}")
            return {
                'has_data': False,
                'error': str(e)
            }


# Test function
if __name__ == "__main__":
    # Example usage
    agent = AISQLAgent()

    # Test query
    result = agent.process_query("Doanh thu dự đoán cho 7 ngày tới là bao nhiêu?")

    if result['success']:
        print("✓ Query thành công!")
        print(f"SQL: {result['sql_query']}")
        print(f"Kết quả: {result['results_count']} dòng")
        print(f"AI Response: {result['ai_response']}")
    else:
        print(f"✗ Lỗi: {result['error']}")
