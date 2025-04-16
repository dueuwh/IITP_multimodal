from transformers import pipeline

class transformer_TER:
    def __init__(self):
        
        self.classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion") # 감정 분석 모델을 사용하여 감정 예측
        
    def run(self, text):
        result = self.classifier(text, , truncation=True, max_length=512)
        emotion = result[0]['label']
        return emotion