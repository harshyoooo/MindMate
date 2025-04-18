# 🧠 MindMate: Your Mental Wellness Companion

**MindMate** is a mental wellness web application designed to help users check in with their emotions and journal their thoughts in a safe, supportive environment.  
It analyzes text, voice, and facial expressions to understand the user's emotional state and provides personalized mental wellness tips.

## ✨ Features

- **Mood Check-In**:  
  Analyze your emotional state by providing text input, uploading a voice recording, or uploading a facial image.
  
- **Journal Entries**:  
  Write and analyze longer journal entries, along with optional audio and image uploads for a more holistic emotional check.

- **Emotion & Sentiment Analysis**:
  - Text-based sentiment and emotion detection
  - Voice emotion analysis from uploaded audio files
  - Facial emotion analysis from uploaded images
  
- **Personalized Suggestions**:  
  Get actionable self-care tips based on your detected mood and emotions.

- **Secure and Anonymous**:  
  No external databases — your entries are kept in-memory and not stored permanently.

## 🚀 How It Works

MindMate uses powerful machine learning models from Hugging Face Transformers:

- **Text Sentiment Analysis**:  
  `distilbert-base-uncased-finetuned-sst-2-english`
  
- **Text Emotion Detection**:  
  `bhadresh-savani/distilbert-base-uncased-emotion`
  
- **Voice Emotion Recognition**:  
  `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
  
- **Facial Emotion Detection**:  
  `dima806/facial_emotions_image_detection`

These models analyze your inputs and generate insights about your emotional state, combined with self-care suggestions.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mindmate.git
   cd mindmate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have the following Python packages installed:
   - `gradio`
   - `pandas`
   - `transformers`
   - `torch`
   - `Pillow`

3. **Run the app:**
   ```bash
   python app.py
   ```

   (Assuming your file is named `app.py`.)

4. **Access it locally** or use the `share=True` option in `demo.launch()` to get a public link.

## 📑 Project Structure

```bash
mindmate/
├── app.py           # Main application code
├── README.md        # Project documentation
├── requirements.txt # Python dependencies
```

## 🛠 How to Use

1. Open the app in your browser.
2. Navigate to the **Mood Check-In** or **Journal** tabs.
3. Input your text and/or upload audio/image files.
4. Click **Analyze** to see your emotional analysis and wellness suggestion!

## 🎯 Future Improvements

- Persistent storage for journal entries (optional user login)
- Visualization of mood over time (charts, graphs)
- More fine-grained suggestions based on combined multi-modal analysis
- Mobile app version

## ❤️ Acknowledgments

- Hugging Face for the amazing pre-trained models.
- Gradio for the easy-to-build web interface.
