from pydantic import BaseModel
from typing import List, Optional

class UserContext(BaseModel):
    currency: str
    locale: str
    financial_level: str

class Summary(BaseModel):
    period: str
    total_income: float
    total_expenses: float
    savings_rate: float

class Category(BaseModel):
    name: str
    amount: float

class Budget(BaseModel):
    name: str
    limit: float
    spent: float

class FinancialInsightsRequest(BaseModel):
    user_context: UserContext
    summary: Summary
    categories: List[Category]
    budgets: Optional[List[Budget]] = []
    goal: str

class Insight(BaseModel):
    type: str
    message: str

class FinancialInsightsResponse(BaseModel):
    insights: List[Insight]
    confidence: float
