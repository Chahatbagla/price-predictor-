import pandas as pd
import re

# Load your dataset
df = pd.read_csv('electronics_product.csv')

# Function to clean price columns
def clean_price(price):
    """
    This function takes a price string, removes non-numeric characters,
    and converts it into a float value for further processing.
    """
    # Convert price to string before applying regex
    price = str(price)
    clean_price = re.sub(r'[^\d.]', '', price)  # Remove any non-numeric characters
    return float(clean_price) if clean_price else None

# Apply the function to both price columns
df['discount_price'] = df['discount_price'].apply(clean_price)
df['actual_price'] = df['actual_price'].apply(clean_price)

# Check for successful conversion
df[['discount_price', 'actual_price']].head()


df.isnull().sum()

