import boto3
import json
import streamlit as st
from langchain import PromptTemplate
from PIL import Image

# AWS clients
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

rek_client = boto3.client('rekognition', region_name='us-east-1')

llm_type = 'titan'

def interactWithLLM(prompt, llm_type):

    if llm_type == 'titan':
        parameters = {
            "maxTokenCount": 512,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
        body = json.dumps({"inputText": prompt, "textGenerationConfig": parameters})
        modelId = "amazon.titan-text-lite-v1"
        accept = "application/json"
        contentType = "application/json"

        response = bedrock_client.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )

        response_body = json.loads(response.get("body").read())

        response_text_titan = response_body.get("results")[0].get("outputText")

        return response_text_titan

def imageAnalyzer(image_bytes):
    response = rek_client.detect_labels(Image={'Bytes': image_bytes})
    labels = response['Labels']
    label_names = ''
    for label in labels:
        name = label['Name']
        confidence = label['Confidence']
        if confidence > 85:
            label_names += name + ","
    return label_names

st.set_page_config(page_title="Image Analysis", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

# Streamlit app
st.title('Image Analysis and Summary Generation')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Convert image to bytes
    image_bytes = uploaded_file.getvalue()

    if image_bytes:
        label_names = imageAnalyzer(image_bytes)
        st.write(label_names)

        prompt_claude = """
        Human: Here are the comma-separated list of labels/objects seen in the image:
        <labels>
        {labels}
        </labels>
        Based on these labels, provide a clear and concise summary of what might be happening in the image.
        Assistant:
        """

        prompt_template_for_summary_generate = PromptTemplate.from_template(prompt_claude)
        prompt_data_for_summary_generate = prompt_template_for_summary_generate.format(labels=label_names)

        response_text = interactWithLLM(prompt_data_for_summary_generate, llm_type)

        st.write('Generated Summary:')
        st.write(response_text)
    else:
        st.write('Error: Uploaded image is empty.')