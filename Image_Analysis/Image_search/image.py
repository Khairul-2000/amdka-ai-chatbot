import requests
import json
from PIL import Image
import base64
from io import BytesIO
import re
from fuzzywuzzy import fuzz
import os
from dotenv import load_dotenv



load_dotenv()


# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found")
        return None

# Function to fetch categories from API
def fetch_categories():
    url = "http://10.10.7.77:3000/api/category/ai"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["success"]:
            return [item["category_name"] for item in data["data"]]
        return []
    except requests.RequestException as e:
        print(f"Error fetching categories: {e}")
        return []

# Function to fetch colors from API
def fetch_colors():
    url = "http://10.10.7.77:3000/api/product/ai-colors"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data["success"]:
            return data["data"]
        return []
    except requests.RequestException as e:
        print(f"Error fetching colors: {e}")
        return []

# Function to call OpenAI GPT Vision API
def analyze_image_with_gpt_vision(image_path, categories, colors):
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return {"category": None, "color": None}

    base64_image = encode_image(image_path)
    if not base64_image:
        return {"category": None, "color": None}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Include category and color lists in the prompt
    prompt = (
        f"Analyze the image and identify the product category and color. "
        f"Choose the category from this list: {json.dumps(categories)}. "
        f"Choose the color from this list: {json.dumps(colors)}. "
        f"Return the result in JSON format as follows: {{\"category\": \"<category>\", \"color\": \"<color>\"}}. "
        f"Ensure the response is valid JSON and contains only the category and color fields. "
        f"If you cannot determine the category or color from the provided lists, use null for that field."
    )

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # print("OpenAI API Response:", json.dumps(result, indent=2))
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            print("Error: Empty content in OpenAI response")
            return {"category": None, "color": None}

        content = re.sub(r'^```json\n|\n```$', '', content).strip()
        
        try:
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, dict) or "category" not in parsed_content or "color" not in parsed_content:
                print("Error: Invalid JSON structure in content:", content)
                return {"category": None, "color": None}
            return parsed_content
        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI response content: {e}")
            print("Raw content:", content)
            return {"category": None, "color": None}

    except requests.RequestException as e:
        print(f"Error calling OpenAI API: {e}")
        return {"category": None, "color": None}

# Function to find closest color match (fallback)
def find_closest_color(target_color, color_list):
    if not target_color or not color_list:
        return None
    
    target_color = target_color.lower().strip()
    
    # Direct match
    for color in color_list:
        if color.lower() == target_color:
            return color
    
    # Fuzzy matching for colors
    best_match = None
    best_score = 0
    similarity_threshold = 80
    
    for color in color_list:
        score = fuzz.partial_ratio(target_color, color.lower())
        if score > best_score and score >= similarity_threshold:
            best_score = score
            best_match = color
    
    return best_match

# Function to find closest category match (fallback)
def find_closest_category(target_category, category_list):
    if not target_category or not category_list:
        return None
    
    target_category = target_category.lower().strip()
    
    # Direct match
    for category in category_list:
        if category.lower() == target_category:
            return category
    
    # Fuzzy matching for categories
    best_match = None
    best_score = 0
    similarity_threshold = 80
    
    for category in category_list:
        score = fuzz.partial_ratio(target_category, category.lower())
        if score > best_score and score >= similarity_threshold:
            best_score = score
            best_match = category
    
    return best_match

# Main function to process image and return JSON
def process_image_search(image_path):
    # Step 1: Fetch dynamic category and color lists
    categories = fetch_categories()
    colors = fetch_colors()
    
    # Step 2: Analyze image using OpenAI GPT Vision with category and color lists
    gpt_response = analyze_image_with_gpt_vision(image_path, categories, colors)
    
    # Step 3: Find closest matches (fallback in case model doesn't use provided lists)
    matched_category = find_closest_category(gpt_response.get("category"), categories)
    matched_color = find_closest_color(gpt_response.get("color"), colors)
    
    # Step 4: Return JSON response
    result = {
        "category": matched_category,
        "color": matched_color
    }
    
    return json.dumps(result, indent=2)

# Example usage
def image_analysis(image_path):
    result = process_image_search(image_path)
    print(result)
    return result