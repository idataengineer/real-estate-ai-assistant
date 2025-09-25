import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from advanced_rag import rag_query, setup_vector_database

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# Initialize RAG database
setup_vector_database()

# Import tools from previous file
from real_estate_agent import calculate_mortgage, property_comparison, affordability_check, tools

class RealEstateAgentWithMemory:
    def __init__(self):
        self.conversation_history = []
        self.user_context = {}  # Store user preferences, budget, etc.
        
    def search_properties(self, query):
        """Search the knowledge base for property information"""
        try:
            result = rag_query(query)
            return {"search_results": result}
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}
    
    def remember_user_info(self, key, value):
        """Store user information for context"""
        self.user_context[key] = value
        return {"message": f"I'll remember that your {key} is {value}"}
    
    def enhanced_tools(self):
        """Extended tool set with RAG search and memory"""
        base_tools = tools.copy()
        
        # Add RAG search tool
        base_tools.append({
            "type": "function",
            "function": {
                "name": "search_properties",
                "description": "Search for property information, market data, or neighborhood details",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query about properties, neighborhoods, or market info"}
                    },
                    "required": ["query"]
                }
            }
        })
        
        # Add memory tool
        base_tools.append({
            "type": "function",
            "function": {
                "name": "remember_user_info",
                "description": "Remember important user information like budget, preferences, family size, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Type of information (budget, family_size, preferred_area, etc.)"},
                        "value": {"type": "string", "description": "The value to remember"}
                    },
                    "required": ["key", "value"]
                }
            }
        })
        
        return base_tools
    
    def chat(self, user_message):
        """Enhanced chat with memory and RAG"""
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Build context from memory
        context = ""
        if self.user_context:
            context = f"User context: {json.dumps(self.user_context, indent=2)}\n"
        
        # System message with context
        system_message = f"""You are an expert real estate agent AI assistant with access to:
- Mortgage and affordability calculators
- Property search and market data
- Conversation memory to personalize responses

{context}

Previous conversation: {self.conversation_history[-5:] if len(self.conversation_history) > 1 else 'None'}

You help clients by:
1. Remembering their preferences and budget
2. Searching for relevant property information
3. Performing calculations when needed
4. Providing personalized recommendations

Be conversational, helpful, and remember details about the client."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Get AI response with tools
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=self.enhanced_tools(),
            tool_choice="auto",
            max_tokens=600
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # Handle tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call appropriate function
                if function_name == "calculate_mortgage":
                    result = calculate_mortgage(**function_args)
                elif function_name == "property_comparison":
                    result = property_comparison(**function_args)
                elif function_name == "affordability_check":
                    result = affordability_check(**function_args)
                elif function_name == "search_properties":
                    result = self.search_properties(function_args["query"])
                elif function_name == "remember_user_info":
                    result = self.remember_user_info(function_args["key"], function_args["value"])
                else:
                    result = {"error": "Unknown function"}
                
                # Add function result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            # Get final response
            final_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=600
            )
            
            response_content = final_response.choices[0].message.content
        else:
            response_content = message.content
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_content})
        
        return response_content

# Interactive chat
if __name__ == "__main__":
    agent = RealEstateAgentWithMemory()
    
    print("🏠 Enhanced Real Estate Agent - With Memory & Search!")
    print("I can remember your preferences and search property data!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Thanks for chatting! Good luck with your real estate journey!")
            break
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent: {response}\n")
        print("-" * 60)