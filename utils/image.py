from PIL import Image
import io
import base64
from typing import Union

def image_to_base64(image_input: Union[Image.Image, str, bytes, io.BytesIO], format="JPEG") -> str:
    """
    Converts an image to a base64-encoded string.

    The input can be:
    - a PIL.Image.Image object,
    - a file path (str),
    - a byte string (bytes),
    - or a BytesIO object.

    Args:
        image_input (Union[PIL.Image.Image, str, bytes, BytesIO]): The image to convert.
        format (str): The format to use when saving the image (default: "JPEG").

    Returns:
        str: The base64-encoded representation of the image.
    """
    # Convert to PIL.Image.Image if not already
    if isinstance(image_input, Image.Image):
        image = image_input
    elif isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, (bytes, io.BytesIO)):
        image = Image.open(io.BytesIO(image_input if isinstance(image_input, bytes) else image_input.read()))
    else:
        raise TypeError("Unsupported image input type. Must be PIL.Image.Image, file path, bytes, or BytesIO.")

    # Save to a buffer and encode
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


if __name__ == '__main__':
    from datasets import load_from_disk
    ds = load_from_disk("data/african-atire")
    image_pil_test = ds['train'][0]['image']
    image_str = image_to_base64(image_pil_test)
    if image_str:
        print("Test 1 passed - image_pil_test ✅")
    else:
        print("Test 1 failed - image_pil_test ❌")
    image_path = "data/african-culture-1/Nigerian Food Dataset/nigerian_food_dataset/test/akara/akara_167.jpg"
    image_str = image_to_base64(image_path)
    if image_str:
        print("Test 2 passed - image_path ✅")
    else:
        print("Test 2 failed - image_path ❌")