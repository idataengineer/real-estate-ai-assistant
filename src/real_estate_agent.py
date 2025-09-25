import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import math

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Tool Functions
def calculate_mortgage(price, down_payment_percent, interest_rate, years):
    """Calculate monthly mortgage payment"""
    try:
        loan_amount = price - (price * down_payment_percent / 100)
        monthly_rate = interest_rate / 100 / 12
        num_payments = years * 12
        
        if monthly_rate == 0:
            monthly_payment = loan_amount / num_payments
        else:
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        return {
            "monthly_payment": round(monthly_payment, 2),
            "total_paid": round(monthly_payment * num_payments, 2),
            "total_interest": round((monthly_payment * num_payments) - loan_amount, 2),
            "loan_amount": round(loan_amount, 2)
        }
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}

def property_comparison(prop1_price, prop2_price, prop1_sqft, prop2_sqft):
    """Compare two properties by price per square foot"""
    try:
        prop1_price_per_sqft = prop1_price / prop1_sqft
        prop2_price_per_sqft = prop2_price / prop2_sqft
        
        better_value = "Property 1" if prop1_price_per_sqft < prop2_price_per_sqft else "Property 2"
        savings = abs(prop1_price_per_sqft - prop2_price_per_sqft) * min(prop1_sqft, prop2_sqft)
        
        return {
            "property1_price_per_sqft": round(prop1_price_per_sqft, 2),
            "property2_price_per_sqft": round(prop2_price_per_sqft, 2),
            "better_value": better_value,
            "potential_savings": round(savings, 2)
        }
    except Exception as e:
        return {"error": f"Comparison error: {str(e)}"}

def affordability_check(annual_income, monthly_debt, home_price):
    """Check if someone can afford a home (28/36 rule)"""
    try:
        monthly_income = annual_income / 12
        max_housing_payment = monthly_income * 0.28  # 28% rule
        max_total_debt = monthly_income * 0.36  # 36% rule
        available_for_housing = max_total_debt - monthly_debt
        
        recommended_max = min(max_housing_payment, available_for_housing)
        
        # Estimate monthly payment (rough calculation)
        estimated_monthly = home_price * 0.005  # Rough estimate
        
        return {
            "can_afford": estimated_monthly <= recommended_max,
            "recommended_max_payment": round(recommended_max, 2),
            "estimated_monthly_payment": round(estimated_monthly, 2),
            "monthly_income": round(monthly_income, 2)
        }
    except Exception as e:
        return {"error": f"Affordability error: {str(e)}"}

# Tool definitions for the AI
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_mortgage",
            "description": "Calculate monthly mortgage payment and total costs",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "Home price in dollars"},
                    "down_payment_percent": {"type": "number", "description": "Down payment as percentage (e.g., 20 for 20%)"},
                    "interest_rate": {"type": "number", "description": "Annual interest rate as percentage (e.g., 6.5 for 6.5%)"},
                    "years": {"type": "number", "description": "Loan term in years (typically 15 or 30)"}
                },
                "required": ["price", "down_payment_percent", "interest_rate", "years"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "property_comparison",
            "description": "Compare two properties by price per square foot",
            "parameters": {
                "type": "object",
                "properties": {
                    "prop1_price": {"type": "number", "description": "Price of first property"},
                    "prop2_price": {"type": "number", "description": "Price of second property"},
                    "prop1_sqft": {"type": "number", "description": "Square footage of first property"},
                    "prop2_sqft": {"type": "number", "description": "Square footage of second property"}
                },
                "required": ["prop1_price", "prop2_price", "prop1_sqft", "prop2_sqft"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "affordability_check",
            "description": "Check if someone can afford a home based on income and debt",
            "parameters": {
                "type": "object",
                "properties": {
                    "annual_income": {"type": "number", "description": "Annual gross income in dollars"},
                    "monthly_debt": {"type": "number", "description": "Monthly debt payments (credit cards, car loans, etc.)"},
                    "home_price": {"type": "number", "description": "Target home price in dollars"}
                },
                "required": ["annual_income", "monthly_debt", "home_price"]
            }
        }
    }
]

def run_agent(user_message):
    """Run the AI agent with tool calling capabilities"""
    
    messages = [
        {
            "role": "system", 
            "content": """You are an expert real estate agent AI assistant. You help clients with:
- Mortgage calculations
- Property comparisons  
- Affordability assessments
- General real estate advice

When users ask about calculations, use the provided tools. Always explain your reasoning and provide helpful context from your real estate expertise."""
        },
        {"role": "user", "content": user_message}
    ]
    
    # First AI call - decide if tools are needed
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=500
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    # If AI wants to use tools
    if message.tool_calls:
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the appropriate function
            if function_name == "calculate_mortgage":
                result = calculate_mortgage(**function_args)
            elif function_name == "property_comparison":
                result = property_comparison(**function_args)
            elif function_name == "affordability_check":
                result = affordability_check(**function_args)
            else:
                result = {"error": "Unknown function"}
            
            # Add function result to conversation
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
        
        # Get final response with tool results
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=500
        )
        
        return final_response.choices[0].message.content
    
    return message.content

if __name__ == "__main__":
    print("🏠 Real Estate AI Agent - Now with Tools!")
    print("Ask me about mortgages, property comparisons, or affordability!\n")
    
    test_questions = [
        "What would be the monthly payment for a $450,000 home with 20% down, 6.5% interest, 30-year loan?",
        "Compare two properties: one costs $450,000 with 2,000 sqft, another costs $620,000 with 2,800 sqft",
        "Can someone with $80,000 annual income and $500 monthly debt afford a $400,000 home?"
    ]
    
    for question in test_questions:
        print(f"🏠 Question: {question}")
        answer = run_agent(question)
        print(f"🤖 Agent: {answer}\n")
        print("-" * 80 + "\n")