"""
CHURN RETENTION AGENT

This is where it becomes an actual "agent" instead of just a script:
- The LLM (Gemini) reasons about the user's question in plain English
- It decides ON ITS OWN when it needs to call predict_churn_risk() to get real data
- It then turns that raw prediction into a human, actionable response

This is the core agent pattern: LLM = brain/reasoning, tool = hands/facts.
The LLM never "guesses" the churn number — it always gets it from our real
trained model via the tool call.

SETUP:
1. Get a free Gemini API key from https://aistudio.google.com/apikey
2. Run: export GEMINI_API_KEY="your_key_here"
3. Run: python churn_agent.py
"""

import os
import json
from google import genai
from google.genai import types
from churn_tool import predict_churn_risk

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 1. Describe our tool to Gemini so it knows when and how to call it.
# This is the "function declaration" — it tells the LLM the tool's name,
# what it does, and what inputs it needs.
predict_churn_tool = types.FunctionDeclaration(
    name="predict_churn_risk",
    description="Predicts a telecom customer's churn risk (probability of leaving) based on their account details, and returns the key risk factors driving that prediction.",
    parameters={
        "type": "object",
        "properties": {
            "gender": {"type": "string", "enum": ["Male", "Female"]},
            "SeniorCitizen": {"type": "integer", "description": "0 or 1"},
            "Partner": {"type": "string", "enum": ["Yes", "No"]},
            "Dependents": {"type": "string", "enum": ["Yes", "No"]},
            "tenure": {"type": "integer", "description": "Months as a customer"},
            "PhoneService": {"type": "string", "enum": ["Yes", "No"]},
            "MultipleLines": {"type": "string", "enum": ["Yes", "No", "No phone service"]},
            "InternetService": {"type": "string", "enum": ["DSL", "Fiber optic", "No"]},
            "OnlineSecurity": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "OnlineBackup": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "DeviceProtection": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "TechSupport": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "StreamingTV": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "StreamingMovies": {"type": "string", "enum": ["Yes", "No", "No internet service"]},
            "Contract": {"type": "string", "enum": ["Month-to-month", "One year", "Two year"]},
            "PaperlessBilling": {"type": "string", "enum": ["Yes", "No"]},
            "PaymentMethod": {"type": "string", "enum": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]},
            "MonthlyCharges": {"type": "number"},
            "TotalCharges": {"type": "number"},
        },
        "required": ["tenure", "Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
    }
)

tools = types.Tool(function_declarations=[predict_churn_tool])
config = types.GenerateContentConfig(
    tools=[tools],
    system_instruction=(
        "You are a customer retention assistant for a telecom company. "
        "When a user describes a customer, extract their details and call "
        "predict_churn_risk to get the real churn probability and risk factors. "
        "NEVER guess the churn number yourself — always use the tool. "
        "Then explain the result in plain language and suggest 2-3 concrete "
        "retention actions based on the specific risk factors returned."
    )
)


def ask_agent(user_message: str) -> str:
    """Send a message to the agent and return its final natural-language answer."""
    contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    candidate = response.candidates[0]

    # 2. Check if the model wants to call our tool
    function_call = None
    for part in candidate.content.parts:
        if part.function_call:
            function_call = part.function_call
            break

    if function_call is None:
        # Model answered directly without needing the tool
        return response.text

    # 3. Actually run our real Python function with the args Gemini extracted
    print(f"[Agent is calling tool: {function_call.name}]")
    args = {k: v for k, v in function_call.args.items()}
    tool_result = predict_churn_risk(args)
    print(f"[Tool returned: {tool_result}]")

    # 4. Send the tool's real result BACK to Gemini so it can write the final answer
    contents.append(candidate.content)
    contents.append(types.Content(
        role="user",
        parts=[types.Part(function_response=types.FunctionResponse(
            name="predict_churn_risk",
            response=tool_result
        ))]
    ))

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )
    return final_response.text


if __name__ == "__main__":
    question = (
        "I have a customer named Riya. She's been with us 2 months, has fiber "
        "optic internet, pays by electronic check, is on a month-to-month "
        "contract, has no online security add-on, and pays 70 dollars a month "
        "with 140 total charges so far. What's her churn risk and what should I do?"
    )
    print("USER:", question)
    print("\nAGENT RESPONSE:\n")
    print(ask_agent(question))
