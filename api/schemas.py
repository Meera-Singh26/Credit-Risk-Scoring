from pydantic import BaseModel, ConfigDict, Field


class Application(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "duration": 24, "amount": 5200, "installment_rate": 3, "present_residence": 2,
        "age": 29, "number_credits": 1, "people_liable": 1,
        "status": "... < 100 DM", "credit_history": "existing credits paid back duly till now",
        "purpose": "car (new)", "savings": "... < 100 DM", "employment_duration": "1 <= ... < 4 years",
        "personal_status_sex": "male : single", "other_debtors": "none", "property": "car or other",
        "other_installment_plans": "none", "housing": "rent", "job": "skilled employee/official",
        "telephone": "no", "foreign_worker": "yes"}})

    duration: int = Field(ge=1, le=120, description="Loan duration in months")
    amount: float = Field(gt=0, le=1_000_000, description="Credit amount")
    installment_rate: int = Field(ge=1, le=4)
    present_residence: int = Field(ge=1, le=4)
    age: int = Field(ge=18, le=100)
    number_credits: int = Field(ge=1, le=10)
    people_liable: int = Field(ge=1, le=10)
    status: str
    credit_history: str
    purpose: str
    savings: str
    employment_duration: str
    personal_status_sex: str | None = Field(None, description="Audit-only; ignored by the model")
    other_debtors: str
    property: str
    other_installment_plans: str
    housing: str
    job: str
    telephone: str
    foreign_worker: str | None = Field(None, description="Audit-only; ignored by the model")


class Reason(BaseModel):
    feature: str
    value: float | int | str | None
    impact: float
    direction: str


class ScoreResponse(BaseModel):
    default_probability: float
    credit_score: int
    risk_band: str
    decision: str
    threshold: float
    model_version: str
    reasons: list[Reason]


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=100)
