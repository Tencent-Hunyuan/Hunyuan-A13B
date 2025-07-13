from transformers import pipeline

class MedicalQA:
    def __init__(self, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

    def answer_question(self, question, context):
        return self.qa_pipeline({'question': question, 'context': context})

if __name__ == '__main__':
    # Initialize the medical QA system
    qa_system = MedicalQA()

    # Provide a context and ask a question
    context = "The patient is a 45-year-old male with a history of hypertension. He presents with a chief complaint of chest pain that started 2 hours ago. The pain is described as a pressure-like sensation in the center of his chest that radiates to his left arm. He also reports shortness of breath and sweating. An EKG was performed and showed ST-segment elevation in the anterior leads, consistent with an acute myocardial infarction."
    question = "What is the patient's diagnosis?"

    # Get the answer from the QA system
    answer = qa_system.answer_question(question, context)
    print(f"Answer: {answer['answer']}")
