from pydantic import BaseModel, Field,field_validator
from typing import Optional, List

class PolicyItem(BaseModel):
    type: str = Field(..., description="type")
    definition: str = Field(..., description="type definition")
    examples: List[str] = Field(..., description="examples")
   
    @field_validator("examples")
    def validate_examples(cls, value):
        if not isinstance(value, list) or len(value) < 1:
            raise ValueError("examples must be a non-empty list")
        return value
    
class TieziSummary(BaseModel):
    summary: str = Field(..., description="summary",min_length=10)
    policy: List[PolicyItem] = Field(..., description="policy")

class EvaluationDataItem(BaseModel):
    text: str = Field(..., description="text")
    label: int = Field(..., description="label")
    context: Optional[List[str]] = Field(None, description="context")
    
    @field_validator("label")
    def validate_label_with_context(cls, value):
        if value not in [0, 1]:
            raise ValueError("label must be 0 or 1")
        return value

class TrainingDataItem(BaseModel):
    text: str = Field(..., description="text")
    label_without_context: int = Field(..., description="label")
    label_with_context: int = Field(..., description="label")
    context: list = Field(..., description="context")
    
    @field_validator("label_with_context")
    def validate_label_with_context(cls, value):
        if value not in [0, 1]:
            raise ValueError("label must be 0 or 1")
        return value
    
    @field_validator("label_without_context")
    def validate_label_without_context(cls, value):
        if value not in [0, 1]:
            raise ValueError("label must be 0 or 1")
        return value