import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="NovaMart RAG Chatbot", page_icon="🛍")
st.title("🛍 NovaMart RAG Chatbot")
st.write("Ask questions about NovaMart's history, policies, operations, and more!")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form"):
    user_input = st.text_input("Your question:", placeholder="e.g., What is NovaMart's return policy?")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    with st.spinner("Thinking..."):
        payload = {"question": user_input}
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                answer = response.json().get("answer", "No answer found.")
            else:
                answer = f"API Error: {response.status_code}"
        except requests.exceptions.ConnectionError:
            answer = "❌ Cannot connect to API. Make sure the API server is running on port 8000."
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"user": user_input, "bot": answer})

if st.session_state.messages:
    st.subheader("Chat History")
    for i, msg in enumerate(reversed(st.session_state.messages)):
        with st.container():
            st.markdown(f"**🙋 You:** {msg['user']}")
            st.markdown(f"**🤖 NovaMart Bot:** {msg['bot']}")
            if i < len(st.session_state.messages) - 1:
                st.divider()
st.sidebar.markdown("### About")
st.sidebar.info("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions about NovaMart using company documents.")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Blessing Phiri](https://www.blessingphiri.dev)")
st.sidebar.markdown("[GitHub Repository](https://github.com/blessing-bester/NovaMart-RAG)")
