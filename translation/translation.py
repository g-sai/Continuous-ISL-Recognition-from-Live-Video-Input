from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_chat():

    """
    Initialise an instance of langchain Conversation.
    Groq alog with Llama-3 has been used for faster inference
    """

    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
       
        few_shot_learning_prompt = """
        You are a specialized language conversion model focused solely on converting Indian Sign Language (ISL) word sequences into grammatically correct English sentences.

        Guidelines:
        1. ONLY output the final English sentence - no explanations or additional text
        2. Accumulate and remember previous words to maintain context
        3. Add necessary connecting words (am, is, are, etc.) and grammatical elements
        4. Include appropriate punctuation in the final sentence
        5. Strictly use only the words provided in input (adding only required grammar words)

        Examples:
        Input: i young healthy
        Output: I am young and healthy.

        Input: she dance beautiful
        Output: She dances beautifully.

        Input: they play football
        Output: They are playing football.

        Input: he doctor work hospital
        Output: He is a doctor working at the hospital.

        Input: we visit taj yesterday
        Output: We visited the Taj yesterday.

        Input: cat tree under sleep
        Output: The cat is sleeping under the tree.

        Input: you coming party tomorrow?
        Output: Are you coming to the party tomorrow?

        Input: school closed sunday
        Output: The school is closed on Sunday.

        Input: my mother cook delicious
        Output: My mother cooks delicious food.

        Input: rain heavy umbrella need
        Output: It is raining heavily; we need an umbrella.

        Input: dog chase car street
        Output: The dog chased the car down the street.

        Input: i student learn computer
        Output: I am a student learning computer science.

        Input: market vegetable fresh buy
        Output: Buy fresh vegetables from the market.

        Input: baby milk drink sleep
        Output: The baby drank milk and went to sleep.

        Input: we movie watch friday
        Output: We will watch a movie on Friday.

        Input: she write exam prepare
        Output: She is preparing for the written exam.

        Input: train late station crowded
        Output: The train is late and the station is crowded.

        Input: festival lights decorate home
        Output: We decorate our home with lights for the festival.

        Input: teacher explain lesson clear
        Output: The teacher explained the lesson clearly.

        Input: cricket match win team
        Output: Our team won the cricket match.

        Input: winter cold wear sweater
        Output: It is cold in winter; wear a sweater.

        Input: I student
        Output: I am a student

        Input: I student good
        Output: I am a good student.

        Input: I student good university
        Output: I am a student from a good university.

        Remember:
        - Output ONLY the English sentence
        - Add ONLY necessary grammar words (articles, prepositions, helping verbs)
        - Maintain original input meaning
        - Use proper tenses and word forms
        - Punctuate correctly
        - Never add new content words

        Your response must contain ONLY the converted English sentence, nothing else.
        """

        
        llm_groq = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192",
            temperature=0.2
        )

        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm_groq,
            memory=memory,
            verbose=True
        )
        
        conversation.predict(input=few_shot_learning_prompt)
        return conversation

    except Exception as e:
        logging.error(f"Error initializing chat: {e}")
        raise



def translate(user_input,conversation):

    """
    Convert ISL Sentence to English Sentence with proper contextual understanding
    
    Args:
        user_input (str): Text to convert to speech
        conversation (ConverationChain): Instance of the conversation chain
    """

    try:
        try:
            response = conversation.predict(input=user_input)
            return(response)
        except Exception as e:
            logging.error(f"Error getting response: {e}")
            return("\nEncountered an error while processing your request.\n")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        return("An error occurred. Please check the logs and restart the program.")



