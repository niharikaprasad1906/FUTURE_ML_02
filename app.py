
# app.py - A Gradio web application for classifying customer support tickets.

import gradio as gr
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- NLTK Data Downloads ---
# Ensure necessary NLTK data is available for text preprocessing
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Global Preprocessing Components ---
# Initialize text processing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Helper Functions for Text Preprocessing ---
def clean_text(text):
    """
    Cleans raw text by converting to lowercase, removing product placeholders,
    special characters, numbers, and extra whitespace.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\{product_purchased\}', '', text) # Escaped for regex
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """
    Applies a series of text preprocessing steps: cleaning, tokenization,
    stopwords removal, and lemmatization.
    """
    # First, clean the text
    text = clean_text(text)
    # Then tokenize it into individual words
    tokens = text.split()
    # Remove common English stopwords and lemmatize words to their base form
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# --- Model Loading ---
# Load the pre-trained classification models and TF-IDF vectorizer.
# These files (ticket_category_model.pkl, ticket_priority_model.pkl, tfidf_vectorizer.pkl)
# must be present in the same directory as this script for the app to run.
try:
    category_model = joblib.load('ticket_category_model.pkl')
    priority_model = joblib.load('ticket_priority_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Successfully loaded classification models and vectorizer.")
except Exception as e:
    print(f"Error loading models. Please ensure 'ticket_category_model.pkl', 'ticket_priority_model.pkl', and 'tfidf_vectorizer.pkl' are in the same directory. Error: {e}")
    # In a production scenario, you might want to exit or provide a fallback UI.

# --- Label Mappings ---
# Define mappings to convert numerical predictions back to human-readable labels.
category_mapping = {
    0: 'Technical issue',
    1: 'Billing inquiry',
    2: 'Refund request',
    3: 'Product inquiry',
    4: 'Cancellation request'
}

priority_mapping = {
    0: 'Low',
    1: 'Medium',
    2: 'High'
}

# --- Prediction and Action Suggestion Functions ---
def get_action_suggestion(category, priority):
    """
    Generates a relevant action suggestion based on the predicted ticket category and priority.
    """
    # This dictionary holds predefined actions for specific category-priority combinations.
    actions = {
        ('Technical issue', 'High'): '🚨 URGENT: Assign to senior tech support. SLA: 1 hour',
        ('Technical issue', 'Medium'): '⚡ Assign to tech support team. SLA: 4 hours',
        ('Technical issue', 'Low'): '📝 Add to tech support queue. SLA: 24 hours',
        ('Billing inquiry', 'High'): '💰 ESCALATE: Involve billing manager. SLA: 2 hours',
        ('Billing inquiry', 'Medium'): '💳 Route to billing team. SLA: 8 hours',
        ('Billing inquiry', 'Low'): '📧 Standard billing response. SLA: 48 hours',
        ('Refund request', 'High'): '💸 PRIORITY: Fast-track refund process. SLA: 2 hours',
        ('Refund request', 'Medium'): '🔄 Process refund request. SLA: 24 hours',
        ('Refund request', 'Low'): '📋 Add to refund queue. SLA: 72 hours',
        ('Cancellation request', 'High'): '⚠️ Immediate retention attempt required',
        ('Cancellation request', 'Medium'): '📞 Retention team follow-up',
        ('Cancellation request', 'Low'): '📧 Process cancellation request',
        ('Product inquiry', 'High'): '📞 Priority - Assign sales/support',
        ('Product inquiry', 'Medium'): 'ℹ️ Product specialist response',
        ('Product inquiry', 'Low'): '📧 Standard product information'
    }
    # Return the specific action or a generic one if no match is found.
    return actions.get((category, priority), '📌 Standard handling required')

def predict_ticket_gradio(subject, description):
    """
    The main prediction function, designed to be called by the Gradio interface.
    It takes a ticket's subject and description, predicts its category and priority,
    and returns a formatted string with results and suggested actions.
    """
    # Combine subject and description into a single text block
    combined_text = f"{subject} {description}"

    # Preprocess the combined text using our defined function
    cleaned_text = preprocess_text(combined_text)

    # Transform the cleaned text into numerical features using the loaded TF-IDF vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict the ticket category and its confidence score
    cat_pred_num = category_model.predict(vectorized_text)[0]
    cat_proba = category_model.predict_proba(vectorized_text)[0]
    cat_confidence = max(cat_proba)

    # Predict the ticket priority and its confidence score
    pri_pred_num = priority_model.predict(vectorized_text)[0]
    pri_proba = priority_model.predict_proba(vectorized_text)[0]
    pri_confidence = max(pri_proba)

    # Convert numerical predictions back to human-readable labels
    category_label = category_mapping[cat_pred_num]
    priority_label = priority_mapping[pri_pred_num]

    # Get a suggested action based on the predicted category and priority
    action_suggestion = get_action_suggestion(category_label, priority_label)

    # Format the prediction results for display in the Gradio interface
    result_output = f"""
    📋 **Predicted Category:** {category_label} ({cat_confidence:.1%} confidence)

    ⚠️ **Predicted Priority:** {priority_label} ({pri_confidence:.1%} confidence)

    💡 **Suggested Action:** {action_suggestion}

    ---
    *Confidence Interpretation:*
    • Low: < 50%
    • Medium: 50-75%
    • High: > 75%
    """
    return result_output

# --- Gradio Interface Definition ---
# This section sets up the interactive web interface using Gradio.
def create_ticket_classifier_interface():
    """
    Builds and returns the Gradio Blocks interface for the ticket classifier.
    """
    # Custom CSS for a slightly improved look and feel of the Gradio app.
    custom_css = """
    <style>
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 0.5em;
        }
        h3 {
            text-align: center;
            color: #34495e;
            font-size: 1.2em;
            margin-top: 0.5em;
            margin-bottom: 1.5em;
        }
        .description {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        .gr-button.primary {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
        }
        .gr-button.primary:hover {
            background-color: #2980b9;
        }
    </style>
    """

    with gr.Blocks(css=custom_css, title="AI Support Ticket Classifier") as demo:
        gr.Markdown("""
        # 🎫 AI-Powered Support Ticket Classifier

        ### Instantly categorize and prioritize customer support requests using machine learning!

        This intelligent system is designed to help your support team by:
        - **Classifying** incoming tickets into 5 common categories (e.g., Technical, Billing, Refund, etc.)
        - **Assigning a priority level** (High, Medium, or Low) based on the content
        - **Suggesting immediate actions** to streamline resolution
        ---
        """)

        with gr.Row():
            with gr.Column():
                subject_input = gr.Textbox(
                    label="📌 Ticket Subject",
                    placeholder="e.g., Cannot login to my account, Product not working",
                    lines=2
                )

                description_input = gr.Textbox(
                    label="📝 Ticket Description",
                    placeholder="Provide a detailed description of the issue here...",
                    lines=6
                )

                classify_button = gr.Button("🔍 Classify Ticket", variant="primary")

                gr.Markdown("### ✨ Need ideas? Try these examples:")
                example_tickets = gr.Examples(
                    examples=[
                        ["URGENT: Server down", "Our entire production server is offline. Customers cannot access the platform. Need immediate help!"],
                        ["Wrong amount charged", "I was charged $99 but my plan is $49. Please refund the difference."],
                        ["Product arrived damaged", "The box was crushed and the product doesn't work. Requesting full refund."],
                        ["Question about features", "Does your software integrate with Salesforce? Need this info for purchasing decision."],
                        ["Cancel my subscription", "Please cancel my annual plan immediately. I no longer need this service."],
                    ],
                    inputs=[subject_input, description_input],
                    label="Click any example to pre-fill the fields"
                )

            with gr.Column():
                output_display = gr.Markdown(
                    label="📊 Classification Result",
                    value="*Submit a ticket to see the AI's prediction!*"
                )

        gr.Markdown("""
        ---
        ### 💡 How to use this tool:
        1. Simply enter the **Subject** and **Description** of a customer support ticket.
        2. Click the "**Classify Ticket**" button.
        3. The system will predict the **Category**, **Priority**, and suggest an **Action** for your team.

        *Note: Model accuracy and response time details here are illustrative. Actual performance depends on your trained models.*
        """)

        # Connect the button click event to our prediction function
        classify_button.click(
            fn=predict_ticket_gradio,
            inputs=[subject_input, description_input],
            outputs=output_display
        )

    return demo

# --- Launch the Gradio Application ---
if __name__ == "__main__":
    app_interface = create_ticket_classifier_interface()
    # Launch the app. `share=True` creates a public link for easy sharing (temporary).
    # For production deployment, consider hosting options like Hugging Face Spaces.
    app_interface.launch(share=True)
