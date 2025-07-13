import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SymptomChecker:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer()
        self.symptom_vectors = self.vectorizer.fit_transform(self.data['symptoms'])

    def find_similar_symptoms(self, user_input):
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.symptom_vectors)
        most_similar_index = similarities.argmax()
        return self.data.iloc[most_similar_index]

if __name__ == '__main__':
    # Create a dummy dataset for testing
    data = {
        'symptoms': [
            'fever, cough, sore throat',
            'headache, nausea, dizziness',
            'stomach pain, diarrhea, vomiting',
            'chest pain, shortness of breath, sweating'
        ],
        'condition': [
            'Common Cold',
            'Migraine',
            'Gastroenteritis',
            'Heart Attack'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv('dummy_symptoms.csv', index=False)

    # Initialize the symptom checker with the dummy dataset
    checker = SymptomChecker('dummy_symptoms.csv')

    # Get user input and find the most similar condition
    user_symptoms = 'fever and cough'
    most_similar_condition = checker.find_similar_symptoms(user_symptoms)
    print(f"Based on your symptoms, you may have: {most_similar_condition['condition']}")
