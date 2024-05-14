import streamlit as st
import home, clean_data
from streamlit_option_menu import option_menu
import os
# from dotenv import load_dotenv
# load_dotenv()

# import home, trending, account, your, about, buy_me_a_coffee
st.set_page_config(
        page_title="Pondering",
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Pondering ',
                options=['Home','Clean Data','Trending'],
                default_index = 0,
                )

        if app == "Home":
            home.app()
        if app == "Clean Data":
            clean_data.app()    
        # if app == "Trending":
        #     trending.app()        
        # if app == 'Your Posts':
        #     your.app()
        # if app == 'about':
        #     about.app()    
        # if app=='Buy_me_a_coffee':
        #     buy_me_a_coffee.app()    
    if 'session_state' not in st.session_state:
        st.session_state.session_state = {}
    run()            

