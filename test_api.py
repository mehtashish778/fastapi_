import requests

# Define the API endpoint
url = "http://127.0.0.1:8000/soap_note/"  # Replace with your FastAPI server URL if different

# Path to the audio file to upload (ensure this is a valid audio file)
audio_file_path = "data/audio_file.wav"  # Replace with the path to your test audio file

# Medical history to send with the request
medical_history = "No significant medical history."

# Prepare the files and form data
files = {
    "audio_file": open(audio_file_path, "rb")
}
data = {
    "medical_history": medical_history
}

try:
    # Make the POST request
    response = requests.post(url, files=files, data=data)

    # Check response status
    if response.status_code == 200:
        print("Response received successfully!")
        print("SOAP Note Output:")
        print(response.json())  # Print the generated SOAP note
    else:
        print(f"Failed with status code: {response.status_code}")
        print("Response content:", response.text)
except Exception as e:
    print("Error occurred:", str(e))
