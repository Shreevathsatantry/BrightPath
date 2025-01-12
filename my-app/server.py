from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from flask_cors import CORS
import re
from PIL import Image
import base64
from io import BytesIO
import time
import requests
import io
from huggingface_hub import InferenceClient
import os
from langchain_community.utilities import SearxSearchWrapper


app = Flask(__name__)
CORS(app)

client = InferenceClient("stabilityai/stable-diffusion-3.5-large-turbo", token="")

llm_llama = Ollama(model="llama3.1:8b-instruct-q8_0")
llm_llava = Ollama(model="ollama run llava:7b-v1.5-q8_0")


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    return response

@app.route("/StoryTeller", methods=["POST"])
def storyTeller():
    input_data = request.get_json()
    input_text = input_data.get("text", "")
    
    response = llm_llama.invoke(f"Tell a children's story based on the given context in 4 paragraphs: {input_text}")

    ImageGen(response.text)
    
    return jsonify({"response": response.text})


def ImageGen(text):
    ParaList = text.split("\n\n")
    for i in range(len(ParaList) - 1):
        PromptImage = llm_llama.invoke(f"Generate a scenario based prompt no more than 20 words to generate an image based on the following context: {ParaList[i]}")
        print(PromptImage.text)
        image = client.text_to_image(PromptImage.text)
        image.save(f"public/Images/Image{i + 1}.png")
    
    client_1 = InferenceClient("stabilityai/stable-diffusion-3.5-large-turbo", token="")
    PromptImage = llm_llama.invoke(f"Generate a scenario based very short prompt to generate an image based on the following context: {ParaList[3]}")
    image = client_1.text_to_image(PromptImage.text)
    image.save(f"public/Image4.png")


@app.route("/QuizBot", methods=["POST"])
def quizBot():
    input_data = request.get_json()
    input_text = input_data.get("text", "")

    text = """
    Generate 10 multiple-choice questions based on the provided story. For each question, include:
    1. The question text.
    2. Four options (labeled a, b, c, d).
    3. skip 2 lines before starting the next question.

    Use this exact format for the response:
    Question 1: [Your question here]
    a) [Option 1]
    b) [Option 2]
    c) [Option 3]
    d) [Option 4]
    Correct Answer: [Letter of the correct answer]

    Repeat for all 10 questions.
    """

    response = llm_llama.invoke(f"in the following format: {text} \n Frame 10 questions and give 4 options with one correct answer on the following story: {input_text}")
    print(response.text)

    quiz_data = parse_quiz_response(response.text)
    print(quiz_data)
    
    return jsonify({"response": quiz_data})

def parse_quiz_response(response):

    questions = re.split(r"(?=Question \d+:)", response.strip())

    parsed_questions = []

    for question in questions:
        lines = question.strip().split("\n")
        if len(lines) >= 6:
            question_text = lines[0].replace("Question \d+: ", "").strip()
            options = [line.strip() for line in lines[1:5]]
            correct_answer = lines[5].replace("Correct Answer: ", "").strip()

            parsed_questions.append({
                "question": question_text,
                "options": options,
                "correctAnswer": correct_answer
            })

    return parsed_questions

@app.route("/LearnBot", methods=["POST"])
def learnBot():
    input_data = request.get_json()
    input_text = input_data.get("text", "")
    image_base64 = input_data.get("image", "")

    image = None
    if image_base64:
        try:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]

            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            print("Error decoding or verifying image:", e)
            return jsonify({"error": "Invalid image data"}), 400

    try:
        if image and input_text:
            prompt = f"Explain this image and the following text in a simple way for children: {input_text}"
            response = llm_llava.invoke([prompt, image])
        elif image:
            prompt = "Explain this image in a simple way for children."
            response = llm_llava.invoke([prompt, image])
        elif input_text:
            prompt = f"Explain this text in a simple way for children: {input_text}"
            response = llm_llava.invoke(prompt)
        else:
            return jsonify({"error": "No valid input provided"}), 400

        return jsonify({"response": response.text})
    except Exception as e:
        print("Error generating response:", e)
        return jsonify({"error": "Failed to generate response"}), 500

@app.route("/AiSuggestionBot", methods=["GET"])
def aiSuggestionBot():

    text = """
    Language Development: Language Development suggestion,
    Physical Development: Physical Development suggestion,
    Cognitive Skills: Cognitive Skills suggestion,
    Communication Skills: Communication Skills suggestion,
    """
    
    response = llm_llama.invoke(f"Give a 4 brief suggestions for the parents on how to improve their childs Language Development, Physical Development, Cognitive Skills, communication skills in the form of: {text}")
    return jsonify({"response": response.text})

def get_downloads_folder():
    """Get the path to the Downloads folder."""
    home = os.path.expanduser("~")
    downloads_folder = os.path.join(home, "Downloads")
    return downloads_folder

@app.route("/VideoAnalyzer", methods=["GET"])
def videoAnalyzer():
    downloads_path = get_downloads_folder()
    file_name = "Video.mp4"
    video_file_name = os.path.join(downloads_path, file_name)

    print(f"Checking for file {file_name} in Downloads folder...")


    while not os.path.exists(video_file_name):
        print(f"Waiting for {file_name} to be available...")
        time.sleep(5)

    print(f"File {file_name} found. Uploading...")

    video_file = llm_llava.upload_file(path=video_file_name)
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        video_file = llm_llava.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    
    print(f'Video processing complete: ' + video_file.uri)

    prompt = """
    Pretend like your talking to the person in the video
    """


    print("Making LLM inference request...")

    response = llm_llava.invoke([prompt, video_file], request_options={"timeout": 600})
    print(response.text)
    os.remove(video_file_name)

    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True)
