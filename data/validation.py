from pydantic import BaseModel, Field


class Validation(BaseModel):
    is_valid: bool = Field(description="if the condition is satisfied")


class UserProfile(BaseModel):
    name: str = Field(description="User name")
    email: str = Field(description="User email")
    subscription: str = Field(description="Subscription type: free or pro")
    user_id: int = Field(description="User id, represented as a number")
    phone: str = Field(description="User phone number")
    language: str = Field(description="User preferred language")


class PhoneCallRequest(BaseModel):
    phone_number: str = Field(description="The user phone number to call")


class PhoneCallTicket(BaseModel):
    agent_name: str = Field(
        description="Name of the shopify agent that answered the call"
    )
    customer_name: str = Field(description="Name of the customer")
    call_summary: str = Field(
        description="Summary of the call between customer and agent"
    )
