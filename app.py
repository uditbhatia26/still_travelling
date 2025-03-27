import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent, AgentExecutor
from config import sys_prompt 
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Web search tool
search = DuckDuckGoSearchRun(name='Search')

# ✅ Initialize LLM with tools
search_llm = ChatGroq(model='llama-3.1-8b-instant', api_key=groq_api_key)
itinerary_llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key)
prompt = sys_prompt
itinerary_generation_chain = prompt | itinerary_llm 
tools = [search]

# 🌍 Travel Planner UI
st.set_page_config(page_title="AI Travel Planner", page_icon="🌍", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center; color:#FF5733;">✈️ AI Travel Planner</h1>
    <p style="text-align:center; font-size:18px;">Let me help you plan your dream trip with AI-powered recommendations! 🌍</p>
    <hr style="border: 1px solid #ddd;">
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("🛫 Travel Preferences")
    starting_location = st.text_input("Starting Location", placeholder="e.g. New Delhi, Mumbai")
    destination = st.text_input("📍 Destination", placeholder="e.g. Paris, Japan")
    budget = st.selectbox(
        "💰 Budget",
        [
            "Economy (💸 < ₹4,000/day)",
            "Standard (💲 ₹4,000 - ₹12,000/day)",
            "Premium (💰 ₹12,000 - ₹25,000/day)",
            "Luxury (✨ ₹25,000+/day)"
        ],
        index=0
    )
    duration = st.slider("📅 Trip Duration (days)", min_value=1, max_value=14, value=5)
    purpose = st.radio("🎯 Purpose", ["Adventure", "Leisure", "Work", "Cultural"], horizontal=True)
    
    # ✅ Button to trigger itinerary generation
    if st.button("Generate Itinerary"):
        st.session_state["generate_itinerary"] = True  # Store flag to trigger LLM

# ✅ Chat Memory for conversation history
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "🌍 Hi! I'm your AI travel planner. Tell me about your trip, and I'll create a customized itinerary!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
# ✅ User input from chat
if user_prompt := st.chat_input(placeholder="Ask me anything about your trip..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("🚀 Fetching the best travel recommendations..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)


if st.session_state.get("generate_itinerary", False):
    with st.chat_message("assistant"):
        with st.spinner("📝 Generating your personalized itinerary..."):
            search_agent = initialize_agent(tools=tools, llm=search_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)
            web_context = search_agent.invoke(input=f"Find tourist spots and hotels for a tourist in {destination}", verbose=True)
            print(web_context)
            itinerary = itinerary_generation_chain.invoke(
                {
                    "starting_location": starting_location,
                    "destination": destination,
                    "duration": duration,
                    "budget": budget,
                    "purpose": purpose,
                    "context": web_context
                }
            )

    st.session_state.messages.append({'role': 'assistant', "content": itinerary})
    st.write(itinerary)
    
    # Reset flag after generation
    st.session_state["generate_itinerary"] = False
