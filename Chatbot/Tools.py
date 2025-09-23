# ---------------------------
# Product search tool
# ---------------------------
from langchain_core.tools import tool
import requests
import json


@tool
def product_search(query: str) -> str:
    """Search for products in the database. Returns all available products for the AI to analyze and select from."""
    print(f"User query: '{query}'")
    
    # API endpoint - fetch all products and let GPT decide which are relevant
    api_url = "http://10.10.7.77:3000/api/product/all"
    search_params = {"limit": 100}
    
    try:
        print(f"Fetching all products from API...")
        
        response = requests.get(
            api_url,
            params=search_params,
            timeout=15,
            headers={'Accept': 'application/json', 'Content-Type': 'application/json'}
        )
        
        print(f"Response status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        products = data.get('data', [])
        print(f"API returned {len(products)} total products for GPT to analyze")
        
        if not data or not products:
            return json.dumps({
                "success": False, 
                "error": "No products found in the database.", 
                "data": []
            })

        # Return all products - let GPT decide which are most relevant
        return json.dumps({
            "success": True, 
            "message": f"Found {len(products)} products. GPT will select the most relevant ones based on your query.",
            "query": query,
            "data": products
        })

    except requests.exceptions.Timeout:
        return json.dumps({"success": False, "error": "Request timed out. Please try again.", "data": []})
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return json.dumps({"success": False, "error": f"Failed to fetch products. Details: {str(e)}", "data": []})
    except Exception as e:
        print(f"Unexpected error: {e}")
        return json.dumps({"success": False, "error": f"An unexpected error occurred: {str(e)}", "data": []})