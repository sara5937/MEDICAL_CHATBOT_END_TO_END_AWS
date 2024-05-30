from setuptools import find_packages, setup

setup(
    name = 'MEDICAL CHATBOT',
    version= '0.0.1',
    author= 'Saran Raj',
    author_email= 'Mr.saranraj777@gmail.com',
    packages= find_packages(),
    install_requires = ["langchain","langchain-community","pypdf","python-dotenv","IPython","streamlit","langchain-chroma","sentence-transformers","langchain-openai"]

)
