# Input field with static key to reliably capture first-time input
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Type your message", label_visibility="collapsed", key="user_input")
with col2:
    send_pressed = st.button("Send")

if send_pressed and user_query.strip():
    st.session_state.chat_history.append(("user", user_query))
    context_df = search_context(user_query)
    answer = call_cohere_chat(user_query, context_df)
    st.session_state.chat_history.append(("bot", answer))
    st.session_state.user_input = ""  # safely clears on next render
