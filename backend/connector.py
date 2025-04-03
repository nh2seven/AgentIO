import ollama
import openai
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama, OpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain


# Class for different connector implementations for various providers
class Connector:
    template = """
    You are a general-purpose AI assistant. 
    Answer the following question based on the context provided.
    If you are not sure of your answer, simply respond with: 
    "I am not sure, try rephrasing the question or asking something else."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    def __init__(self, provider="ollama", model=None, api_key=None):
        self.provider = provider.lower()
        self.model = model or self.default_model(provider)
        self.api_key = api_key
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        self.llm = self.init_llm()

    # Function to get the default model based on the provider
    def default_model(self, provider):
        defaults = {
            "ollama": "mistral",
            "openai": "gpt-4",
            "huggingface": "meta-llama/Meta-Llama-3-8B"
        }
        return defaults.get(provider, "mistral")

    # Function to initialize the LLM based on the provider
    def init_llm(self):
        if self.provider == "ollama":
            return Ollama(model=self.model)
        
        elif self.provider == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required.")
            return OpenAI(model=self.model, openai_api_key=self.api_key)
        
        elif self.provider == "huggingface":
            generator = pipeline("text-generation", model=self.model)
            return HuggingFacePipeline(pipeline=generator)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # Function to query the LLM with context and question
    def query(self, context, question):
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return llm_chain.run(context=context, question=question)
