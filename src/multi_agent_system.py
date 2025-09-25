import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from advanced_rag import rag_query, setup_vector_database
from real_estate_agent import calculate_mortgage, property_comparison, affordability_check

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Initialize RAG
setup_vector_database()

class ResearchAgent:
    """Specializes in market research and data analysis"""
    
    def analyze_market(self, query):
        """Research market trends and data"""
        research_prompt = f"""
        As a real estate market research specialist, analyze this query: {query}
        
        Provide insights on:
        - Market trends
        - Investment potential
        - Risk factors
        - Opportunities
        
        Base your analysis on available data.
        """
        
        # Get market data from RAG
        market_data = rag_query(f"market trends investment {query}")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a real estate market research specialist focused on data-driven analysis."},
                {"role": "user", "content": f"{research_prompt}\n\nAvailable data: {market_data}"}
            ],
            max_tokens=400
        )
        
        return response.choices[0].message.content

class FinancialAgent:
    """Specializes in financial analysis and calculations"""
    
    def financial_analysis(self, price, income, down_payment_percent=20, interest_rate=6.5):
        """Comprehensive financial analysis"""
        
        # Use existing tools
        mortgage_calc = calculate_mortgage(price, down_payment_percent, interest_rate, 30)
        affordability = affordability_check(income, 0, price)  # Assuming no other debt for simplicity
        
        analysis_prompt = f"""
        As a financial advisor, provide comprehensive analysis for:
        - Property price: ${price:,}
        - Buyer income: ${income:,}
        - Mortgage details: {mortgage_calc}
        - Affordability check: {affordability}
        
        Provide recommendations on:
        1. Monthly budget impact
        2. Long-term financial implications
        3. Alternative scenarios
        4. Financial recommendations
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a financial advisor specializing in real estate purchases."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=400
        )
        
        return response.choices[0].message.content

class CustomerAgent:
    """Handles customer interactions and coordinates other agents"""
    
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.financial_agent = FinancialAgent()
        self.conversation_memory = []
    
    def coordinate_response(self, user_message):
        """Decide which agents to involve and coordinate response"""
        
        # Determine what type of help is needed
        coordinator_prompt = f"""
        User message: "{user_message}"
        
        Determine if this requires:
        1. Market research (trends, investment, neighborhoods)
        2. Financial analysis (affordability, mortgages, budgets)  
        3. General real estate advice
        4. Property search
        
        Respond with JSON: {{"needs": ["research", "financial", "search"], "priority": "primary_need"}}
        """
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a coordinator who determines what type of real estate help is needed."},
                {"role": "user", "content": coordinator_prompt}
            ],
            max_tokens=100
        )
        
        try:
            needs = json.loads(response.choices[0].message.content)
        except:
            needs = {"needs": ["search"], "priority": "search"}
        
        # Collect responses from needed agents
        agent_responses = {}
        
        if "research" in needs["needs"]:
            agent_responses["research"] = self.research_agent.analyze_market(user_message)
        
        if "financial" in needs["needs"]:
            # Extract price and income if mentioned
            try:
                # Simple extraction - in production would be more sophisticated
                words = user_message.split()
                price = 500000  # Default
                income = 80000   # Default
                
                for i, word in enumerate(words):
                    if '$' in word or 'price' in word.lower():
                        try:
                            price = int(''.join(filter(str.isdigit, word)))
                        except:
                            pass
                    if 'income' in word.lower() and i < len(words)-1:
                        try:
                            income = int(''.join(filter(str.isdigit, words[i+1])))
                        except:
                            pass
                
                agent_responses["financial"] = self.financial_agent.financial_analysis(price, income)
            except:
                agent_responses["financial"] = "Financial analysis requires property price and income information."
        
        if "search" in needs["needs"]:
            agent_responses["search"] = rag_query(user_message)
        
        # Synthesize final response
        synthesis_prompt = f"""
        User asked: "{user_message}"
        
        Agent responses:
        {json.dumps(agent_responses, indent=2)}
        
        Provide a comprehensive, helpful response that synthesizes insights from all agents.
        Be conversational and focus on what's most important for the user.
        """
        
        final_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful real estate assistant providing comprehensive guidance."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=500
        )
        
        return final_response.choices[0].message.content

# Interactive multi-agent system
if __name__ == "__main__":
    customer_agent = CustomerAgent()
    
    print("🏠 Multi-Agent Real Estate System!")
    print("🔬 Research Agent: Market analysis and trends")
    print("💰 Financial Agent: Calculations and budgeting")  
    print("🤖 Customer Agent: Coordinates everything for you")
    print("\nType 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thanks for using our multi-agent system!")
            break
        
        print("\n🤖 Coordinating agents...")
        response = customer_agent.coordinate_response(user_input)
        print(f"\n🏠 Complete Analysis: {response}\n")
        print("-" * 80)