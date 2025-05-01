from PIL import Image
from openai import AzureOpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import requests

from utils.image import image_to_base64

# Load environment variables from .env file
load_dotenv()

# Set the API key from the environment variable
API_KEY = os.getenv("AZURE_OPENAI_KEY")
if API_KEY is None:
    raise ValueError("API key not found. Please set the AZURE_OPENAI_API_KEY environment variable.")

def describe_image(prompt, *, image=None, image_format="JPEG", image_url=None, label=None, show_image=True):
    """
    Sends a request to Azure OpenAI's GPT-4o model with a prompt and an image.

    The image can be provided in two ways:
    - As a local image loaded with PIL (`image`)
    - As an accessible image URL (`image_url`)

    Args:
        prompt (str): The prompt to send to the model (e.g., "Describe this image").
        image (PIL.Image.Image, optional): The PIL image to send (converted to base64).
        image_format (str, optional): The format of the image (e.g., "JPEG", "PNG").
            This is used to construct the correct MIME type in the data URL.
        image_url (str, optional): A direct URL to an image hosted online.
        label (str, optional): A label for the image.
        show_image (bool, optional): Whether to display the image using matplotlib.
            Defaults to True. If False, the image will not be displayed.

    Returns:
        str or None: The model's response (image description), or None if an error occurred.
    """
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2024-05-01-preview",
        azure_endpoint="https://instancehackatonpionners01.openai.azure.com",
        azure_deployment="gpt-4o-pionners27"
    )

    try:
        if label is not None:
            prompt = f"{prompt} The label of the image is {label}."
        # Build the image payload
        if image_url is not None:
            # Case 1: Image provided via URL
            image_payload = {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        elif image is not None:
            # Case 2: Local image provided via PIL, convert to base64
            base64_str = image_to_base64(image, format=image_format)
            mime_type = f"image/{image_format.lower()}"
            image_payload = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}
            }
       
        else:
            raise ValueError("Either `image` or `image_url` must be provided.")

        # Send the request to the API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that describes images accurately."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    image_payload
                ]}
            ],
            max_tokens=100
        )
        
        description = response.choices[0].message.content
        
        # Display the image if requested
        if show_image:
            if image is not None:
                plt.imshow(image)
            else:
                img = Image.open(requests.get(image_url, stream=True).raw)
                plt.imshow(img)
            plt.axis('off')
            plt.show()

        # Return the model's answer
        return description

    except Exception as e:
        print(f"Error: {e}")
        return None
    
if __name__ == '__main__':
    # Example usage
    image_path = "data/others/Nigerian Food Dataset/nigerian_food_dataset/test/akara/akara_167.jpg"
    image = Image.open(image_path)
    prompt = "Dis moi ce que tu vois sur l'image."
    
    description = describe_image(prompt, image=image, show_image=True)
    print(description)
