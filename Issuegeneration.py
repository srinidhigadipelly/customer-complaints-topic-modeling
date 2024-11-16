# IMPORT LIBRARIES
import boto3
import json
import pandas as pd

# Simplified prompt to instruct the model
prompt_data1 = """
You are an AI designed to extract the main issue from the text. Please summarize the issue in 2-3 words, strictly without any unnecessary details or phrases like "The main issue is that".
"""

# Load the dataset (replace 'your_dataset.csv' with the actual path to your dataset)
df = pd.read_csv('')  # Dataset with 'body_new' column
print("Dataset loaded successfully.")

# Select only the first 100 emails from the dataset
df_10_emails = df.head(100)

# Create a Bedrock Runtime client in the AWS Region
bedrock = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id='your_access_key_id',
    aws_secret_access_key='your_secret_access_key',
    region_name='your_region_name'
)

# Function to format the request and get the response from Llama3
def extract_issue(email_text):
    # Truncate email to first 512 characters to avoid lengthy input
    truncated_email = email_text[:512]

    # Embed the message in Llama3's prompt format
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt_data1}

<|eot_id|><|start_header_id|>user<|end_header_id|>{truncated_email}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    # Format the request payload using the model's native structure
    payload = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.6
    }

    # Encode and send the request
    body = json.dumps(payload)

    # Set the model ID for Llama3
    model_id = "meta.llama3-70b-instruct-v1:0"

    # Retry mechanism
    for attempt in range(3):  # Try up to 3 times to get a valid response
        try:
            # Invoke the model and get the response
            response = bedrock.invoke_model(
                body=body,
                modelId=model_id,
                accept="application/json",
                contentType="application/json"
            )

            # Decode the response
            response_body = json.loads(response.get("body").read())
            # Extract the generated issue from the response
            response_text = response_body['generation']

            # Clean the response and ensure it contains valid text
            issue = response_text.strip()

            # Validate the issue
            if issue and issue not in ["مْ", ""]:  # Add other unwanted responses as needed
                return issue.split('.')[0] + '.'  # Ensure only one sentence is returned

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

    # Fallback if all attempts fail
    return "Unable to extract issue"

print("Starting issue extraction...")

# Create a new DataFrame with only two columns: original text and extracted issue
df_issues = pd.DataFrame({
    'email': df_10_emails['body_new'],  # Replace 'body_new' with the correct column name if needed
    'issue': df_10_emails['body_new'].apply(extract_issue)
})

print("Issue extraction complete")

# Save the result to a new CSV file
df_issues.to_csv('customer_service_email_issues_dataset.csv', index=False)

print("Dataset saved as 'customer_service_email_issues_dataset.csv'.")
