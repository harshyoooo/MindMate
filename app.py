import gradio as gr
import pandas as pd
from transformers import pipeline
from datetime import datetime
import tempfile
from PIL import Image

def load_models():
    # Hugging Face token is typically set as an environment variable in Spaces
    # Use os.environ.get("HUGGINGFACE_TOKEN") if needed, but anonymous access should work
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=6)
    voice_emotion_model = pipeline("audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    face_emotion_model = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

    test_text = "I'm feeling down today."
    print("Testing models:")
    print(sentiment_model(test_text))
    print(f"Dominant emotion: {max(emotion_model(test_text)[0], key=lambda x: x['score'])['label']}")

    return sentiment_model, emotion_model, voice_emotion_model, face_emotion_model

sentiment_model, emotion_model, voice_emotion_model, face_emotion_model = load_models()

entries_df = pd.DataFrame(columns=["date", "mood", "journal", "sentiment", "emotion", "voice_emotion", "face_emotion"])

def save_entries():
    pass

def analyze_text(text):
    if text:
        sentiment_result = sentiment_model(text)[0]
        sentiment_label = sentiment_result['label']
        emotion_result = emotion_model(text)[0]
        dominant_emotion = max(emotion_result, key=lambda x: x['score'])['label']
        return sentiment_label, dominant_emotion
    return None, None

def analyze_voice(audio_path):
    if audio_path:
        try:
            with open(audio_path, "rb") as f:
                with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                    temp_audio.write(f.read())
                    temp_audio.seek(0)
                    voice_result = voice_emotion_model(temp_audio.name)[0]
                    dominant_voice_emotion = voice_result['label']
                    return dominant_voice_emotion
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
    return None

def analyze_face(image):
    if image:
        try:
            img = Image.open(image)
            face_result = face_emotion_model(img)[0]
            dominant_face_emotion = face_result['label']
            return dominant_face_emotion
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    return None

def get_sentiment_from_emotion(emotion):
    positive_emotions = ['happy', 'joy', 'surprise']
    negative_emotions = ['sad', 'angry', 'fear', 'disgust']
    if emotion.lower() in positive_emotions:
        return 'POSITIVE'
    elif emotion.lower() in negative_emotions:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def suggest_tips(sentiment, emotion):
    tips = {
        "POSITIVE": {
            "joy": "You're radiating positivity! Keep it up with a gratitude journal or a short walk.",
            "default": "Great to see you feeling good! Try some light exercise or mindfulness."
        },
        "NEGATIVE": {
            "sadness": "It’s okay to feel down. Try a guided meditation or chat with a friend.",
            "anger": "Feeling frustrated? Take deep breaths or journal your thoughts.",
            "fear": "Feeling anxious? Try a 5-minute breathing exercise or soothing music.",
            "default": "Tough moment? Consider reading or a warm bath."
        }
    }
    sentiment_tips = tips.get(sentiment, tips["NEGATIVE"])
    return sentiment_tips.get(emotion, sentiment_tips["default"])

with gr.Blocks(title="MindMate", css=".container {padding: 20px; background-color: #f0f4f8; border-radius: 10px;} .input-group {margin-bottom: 15px;}") as demo:
    gr.Markdown("### 🧠 MindMate: Your Mental Wellness Companion")
    gr.Markdown("A safe space to check in with your emotions and journal your thoughts.")

    with gr.Tab("Mood Check-In"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Text Input")
                mood_input = gr.Textbox(label="How are you feeling today? Share a few words or a sentence.", value="")
            with gr.Column(scale=1):
                gr.Markdown("Media Inputs (Optional)")
                audio_input = gr.File(label="Upload an audio file", type="filepath")
                image_input = gr.File(label="Upload an image", type="filepath")
        mood_output = gr.Textbox(label="Analysis", interactive=False)
        mood_button = gr.Button("Analyze Mood")

        def mood_analyze(mood, audio, image):
            sentiment, emotion = analyze_text(mood)
            voice_emotion = analyze_voice(audio)
            face_emotion = analyze_face(image)

            if sentiment and emotion:
                suggestion = suggest_tips(sentiment, emotion)
            elif voice_emotion:
                sentiment = get_sentiment_from_emotion(voice_emotion)
                suggestion = suggest_tips(sentiment, voice_emotion)
            elif face_emotion:
                sentiment = get_sentiment_from_emotion(face_emotion)
                suggestion = suggest_tips(sentiment, face_emotion)
            else:
                return "Please provide at least one input (text, audio, or image)."

            new_entry = pd.DataFrame({
                "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "mood": [mood if mood else ""],
                "journal": [""],
                "sentiment": [sentiment if sentiment else ""],
                "emotion": [emotion if emotion else ""],
                "voice_emotion": [voice_emotion if voice_emotion else ""],
                "face_emotion": [face_emotion if face_emotion else ""]
            })
            global entries_df
            entries_df = pd.concat([entries_df, new_entry], ignore_index=True)
            save_entries()

            output = ""
            if sentiment and emotion:
                output += f"Text Sentiment: {sentiment}\nText Emotion: {emotion}\n"
            if voice_emotion:
                output += f"Voice Emotion: {voice_emotion}\n"
            if face_emotion:
                output += f"Face Emotion: {face_emotion}\n"
            output += f"Suggestion: {suggestion}"
            return output

        mood_button.click(fn=mood_analyze, inputs=[mood_input, audio_input, image_input], outputs=mood_output)

    with gr.Tab("Journal"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("Text Input")
                journal_input = gr.Textbox(label="Write your thoughts here:", lines=10, value="")
            with gr.Column(scale=1):
                gr.Markdown("Media Inputs (Optional)")
                audio_input_journal = gr.File(label="Upload an audio file", type="filepath")
                image_input_journal = gr.File(label="Upload an image", type="filepath")
        journal_output = gr.Textbox(label="Analysis", interactive=False)
        journal_button = gr.Button("Analyze Journal")

        def journal_analyze(journal, audio, image):
            sentiment, emotion = analyze_text(journal)
            voice_emotion = analyze_voice(audio)
            face_emotion = analyze_face(image)

            if sentiment and emotion:
                suggestion = suggest_tips(sentiment, emotion)
            elif voice_emotion:
                sentiment = get_sentiment_from_emotion(voice_emotion)
                suggestion = suggest_tips(sentiment, voice_emotion)
            elif face_emotion:
                sentiment = get_sentiment_from_emotion(face_emotion)
                suggestion = suggest_tips(sentiment, face_emotion)
            else:
                return "Please provide at least one input (text, audio, or image)."

            new_entry = pd.DataFrame({
                "date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "mood": [""],
                "journal": [journal if journal else ""],
                "sentiment": [sentiment if sentiment else ""],
                "emotion": [emotion if emotion else ""],
                "voice_emotion": [voice_emotion if voice_emotion else ""],
                "face_emotion": [face_emotion if face_emotion else ""]
            })
            global entries_df
            entries_df = pd.concat([entries_df, new_entry], ignore_index=True)
            save_entries()

            output = ""
            if sentiment and emotion:
                output += f"Text Sentiment: {sentiment}\nText Emotion: {emotion}\n"
            if voice_emotion:
                output += f"Voice Emotion: {voice_emotion}\n"
            if face_emotion:
                output += f"Face Emotion: {face_emotion}\n"
            output += f"Suggestion: {suggestion}"
            return output

        journal_button.click(fn=journal_analyze, inputs=[journal_input, audio_input_journal, image_input_journal], outputs=journal_output)

demo.launch(share=True)
