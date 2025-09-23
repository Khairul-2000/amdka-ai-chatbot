from langchain_core.messages import  AIMessage, SystemMessage
from config import AIState, model




# ---------------------------
# Agent Node
# ---------------------------
def agent_node(state: AIState):
    system_prompt = SystemMessage(content="""
    You are an intelligent shopping assistant with advanced product analysis capabilities and conversation memory.
    
    CRITICAL: You MUST ALWAYS respond in the following JSON format ONLY. No exceptions!
    
    REQUIRED RESPONSE FORMAT:
    {
        "message": "your response message here",
        "products": ["product_id_1", "product_id_2"] or null
    }
    
    STRICT JSON RULES:
    - Your response MUST be valid JSON that starts with { and ends with }
    - If recommending products, put their exact IDs from the database in the "products" array
    - If no products to recommend, set "products": null
    - The "message" field contains your helpful response with product details
    - DO NOT include product IDs in the message text - only in the "products" array
    - NEVER respond with plain text - ALWAYS respond with this exact JSON structure
    - Test your JSON before responding to ensure it's valid
    
    MEMORY & CONTEXT:
    - You can remember previous conversations and user preferences from earlier messages.
    - Reference past interactions when relevant (e.g., "Based on your earlier interest in red products...")
    - Build on previous searches and recommendations to provide better assistance.
    - If a user mentions something they liked or disliked before, remember and use that information.
    
    PRODUCT SEARCH INSTRUCTIONS:
    - Use the product_search tool to get ALL available products from the database.
    - YOU must intelligently analyze the full product list and select the most relevant products based on the user's specific requirements.
    - Pay close attention to user criteria like colors, sizes, product types, and descriptions.
    - When user asks for "red products in size M", find products that have BOTH "red" (or "Red") in their colors array AND "M" in their sizes array.
    - Analyze all product attributes: product_name, description, colors, sizes, price, offer_price.
    - Present only the top 2-3 most relevant products that exactly match the user's criteria.
    - If no products match the exact criteria, suggest the closest alternatives.
    - Only call the product_search tool when you need fresh product data or when the user asks for a new/different search.
    
    EXAMPLE RESPONSES:
    
    Example 1 - Product Recommendation:
    {
        "message": "I found great red outfits for you:\n\n1. **Premium Red Shirt** - $800 (was $1200), available in M/L/XL\n2. **Casual Red Shoes** - $1300, available in M/L\n\nBoth items are currently on sale!",
        "products": ["cmfrt227w0002vhfsh8ez0sic", "cmfrt227x0007vhfst7knoqqn"]
    }
    
    Example 2 - General Conversation:
    {
        "message": "Hello! I'm here to help you find the perfect products. What are you looking for today?",
        "products": null
    }
    
    MESSAGE FORMATTING FOR PRODUCTS:
    - Format products nicely with name, price, colors, sizes, description
    - Use numbered lists for multiple products
    - DO NOT include product IDs in the message - they belong only in the "products" array
    
    CONVERSATION FLOW:
    - Be conversational and remember what the user has asked about before.
    - If they're asking follow-up questions, provide helpful additional information without unnecessary tool calls.
    - Personalize responses based on conversation history.
    - Be precise and thorough in your analysis - the user is counting on your intelligence to find the right products.
    - Always include product IDs in the "products" array when recommending specific products.
    
    FINAL REMINDER: Every single response must be valid JSON with "message" and "products" fields. No exceptions!
    """)

    messages = [system_prompt] + list(state["messages"])
    response = model.invoke(messages)
    return {"messages": [response]}


# ---------------------------
# Tool Output Formatter (Product API)
# ---------------------------
def tool_output_node(state: AIState):
    """Convert product API response into a structured format for the AI to process."""
    last_message = state["messages"][-1]

    if not hasattr(last_message, "content"):
        return {}

    try:
        import json
        data = json.loads(last_message.content) if isinstance(last_message.content, str) else last_message.content

        # Handle error cases
        if not data.get("success", True):
            error_msg = data.get("error", "Unknown error occurred")
            summary = f"Product search error: {error_msg}. Please try again or ask for help with something else."
        elif not data.get("data") or len(data.get("data", [])) == 0:
            summary = "No products found matching your criteria. The product database returned no results."
        else:
            products = data["data"]
            # Format product information for AI processing
            product_info = []
            
            for p in products:
                price_info = f"${p.get('offer_price', p.get('price', 'N/A'))}"
                if p.get("offer_price") and p.get("price") and p["offer_price"] != p["price"]:
                    price_info = f"${p['offer_price']} (was ${p['price']})"
                
                product_info.append({
                    "id": p.get('id'),
                    "name": p.get('product_name', 'Unknown Product'),
                    "price": price_info,
                    "colors": p.get('colors', []),
                    "sizes": p.get('sizes', []),
                    "description": p.get('description', 'No description available')
                })

            summary = f"Found {len(products)} products. Here are the available products:\n" + json.dumps(product_info, indent=2)

    except json.JSONDecodeError as e:
        summary = f"Error parsing product data: The server returned invalid data. Please try again later."
    except Exception as e:
        summary = f"Error processing product data: {str(e)}"

    return {"messages": [AIMessage(content=summary)]}
