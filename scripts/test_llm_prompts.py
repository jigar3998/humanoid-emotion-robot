# scripts/test_llm_prompts.py
import ollama

test_cases = [
    {'label': 'Person crying after breakup',
     'face': 'sad', 'voice': 'sad', 'crying': True,
     'speech': "We broke up after 3 years. I don't know what to do."},
    {'label': 'Failed exam',
     'face': 'sad', 'voice': 'neutral', 'crying': False,
     'speech': "I failed my exam again."},
    {'label': 'Got promoted',
     'face': 'happy', 'voice': 'happy', 'crying': False,
     'speech': "I just got promoted!"},
    {'label': 'Anxious about presentation',
     'face': 'fear', 'voice': 'neutral', 'crying': False,
     'speech': "I have a big presentation tomorrow and I'm terrified."},
    {'label': 'Boss took credit',
     'face': 'angry', 'voice': 'angry', 'crying': False,
     'speech': "My boss took credit for my entire project."},
    {'label': 'Silent but sad',
     'face': 'sad', 'voice': 'sad', 'crying': False,
     'speech': ''},
]

for case in test_cases:
    print(f"\n{'='*55}")
    print(f"Scenario: {case['label']}")
    print(f"Said: {case['speech'] or '(nothing)'}")

    prompt = f"""You are a compassionate humanoid robot.
Face shows: {case['face']}, voice tone: {case['voice']}.
{'Person is crying.' if case['crying'] else ''}
Person said: "{case['speech']}"
Respond with genuine empathy in 2-3 sentences.
Never use generic phrases like "I understand" or "I'm here for you"."""

    resp = ollama.chat(
        model='llama3.2:3b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    print(f"Response: {resp['message']['content']}")
