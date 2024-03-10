import streamlit as st
import json
import boto3
from crewai import Agent, Task, Crew, Process
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_secrets(parameter):
    session = boto3.session.Session(profile_name='javacoder')
    client = session.client('ssm', region_name='us-east-1')
    secret_key = parameter
    response = client.get_parameter(Name=secret_key, WithDecryption=True)
    return json.dumps(response['Parameter']['Value'])

st.subheader("Gen-Marketer")

api_gemini = get_secrets('google_api_token')
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", verbose=True, temperature=0.1, api_gemini=api_gemini
)

# print(type(api_gemini))
# print(api_gemini)
# exit()

market_researcher = Agent(
role = 'Market Researcher',
goal='Research new and emerging trends in pet products industry in Germany',
backstory = 'You are a market researcher in the pet product industry',
verbose= True,
allow_delegation= False,
    #llm = an llm of your choice can be taken, but default is chatgpt or gpt 3.5
#llm= ChatOpenAI(model_name="gpt-3.5", temperature=0.7)#ollama_openhermes
    llm=llm
)

campaign_creator =  Agent(
role = 'Marketing Campaign Creator',
goal='Come up with 3 interesting marketing campaign ideas in the pet product industry based on market research insights',
backstory = 'You are a marketing campaign planner in the pet product industry',
verbose= True,
allow_delegation= False,
    #llm = an llm of your choice can be taken, but default is chatgpt or gpt 3.5
llm=llm

    
)

digital_marketer =  Agent(
role = 'Digital Marketing Content Creator',
goal='Come up with 2 or 3 interesting advertisement ideas for marketing on digital platforms such as Youtube, Instagram amd Tiktok along with script for each marketing campaign',
backstory = 'You are a marketing marketer specialising in performance marketing in the pet product industry',
verbose= True,
allow_delegation= False,
    #llm = an llm of your choice can be taken, but default is chatgpt or gpt 3.5
#llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7) #ollama_openhermes
    llm=llm

)


prompt = st.text_area("What Market Research Task would You like me to do Today?")
task1 = Task(description=prompt, agent=market_researcher, expected_output="A bullet list summary of the top 5")

prompt = st.text_area("What Marketing Campaigns would You like me to come up with Today?")
task2 = Task(description=prompt, agent=campaign_creator, expected_output="A bullet list summary of the top 5")

prompt = st.text_area("What Digital Marketing Content would You like me to generate Today?")
task3 = Task(description=prompt, agent=digital_marketer, expected_output="A bullet list summary of the top 5")

#1 = Task(description='Come up with 3 marketing trends in the pet industry in the food department, grooming department and toys department', agent=market_researcher)
#2 = Task(description='Come up with marketing campaign ideas based on market research trends', agent=campaign_creator)
#3 = Task(description='Produce digital marketing content related to marketing trends', agent=digital_marketer)

# Create crew
crew = Crew(
agents=[market_researcher,campaign_creator,digital_marketer],
tasks=[task1,task2,task3],
verbose = 2,
process = Process.sequential
)


# query answering
if st.button("Generate"):
        #if prompt:
            # call pandas_ai.run(), passing dataframe and prompt
    with st.spinner("Generating response..."):
                #if 'show' or 'plot' or 'graph' in prompt:
                 #   plot = pandas_ai.run(option, prompt) #option.chat(prompt)
                  #  st.pyplot(plot.fig)
                #else:
                #option = SmartDataframe(option, config={"llm": llm})
        st.write(crew.kickoff())
        st.write(task1.output)
        st.write(task2.output)
        st.write(task3.output)
         #(option.chat(prompt)) #(pandas_ai.run(option, prompt))

        #else:
        #    st.warning("Please enter a prompt.")