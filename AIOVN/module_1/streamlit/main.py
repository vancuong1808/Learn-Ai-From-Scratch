import streamlit as st

st.title("Streamlit")
st.divider()

st.header("Learn streamlit from scratch")
st.subheader("1. Basic")
st.text("---To do----")
st.caption("Note: *****")
st.divider()

st.markdown("# This Markdown Heading")
st.markdown("""
            1. Machine Learning
            2. Deep Learning
            """)
st.markdown('$ \sqrt{2x}  $')
st.divider()

st.latex(' \sqrt{2x} ')
st.divider()

st.write("Learn streamlit")
st.write("# Learn streamlit")
st.write("[ streamlit ](http://)")
st.write('$ \sqrt{2x}  $')
st.divider()

st.code("""
        import streamlit as st

        st.title("Streamlit")
        st.divider()

        st.header("Learn streamlit from scratch")
        st.subheader("1. Basic")
        st.text("---To do----")
        st.caption("Note: *****")
        st.divider()
        """)
st.divider()

def get_name():
    return "Cuong"

def get_age():
    return 20

with st.echo():
    st.write("Test echo def")

    name = get_name()
    age = get_age()
    st.write(name, age)
st.divider()

st.logo("./streamlit/logo.png")
st.image("./streamlit/dog.jpeg", caption="Viet Toan")
st.audio("./streamlit/audio.mp4")
st.video("./streamlit/video.mp4")
st.divider()

def get_fullname():
    return True

agree = st.checkbox("I agree", on_change=get_fullname)
if agree:
    st.write(agree)

status = st.radio('Your choice:', ["Yes", "No"], captions=['go out', 'at home'])
st.write(status)
st.divider()

selected = st.selectbox('Your choice', ["Yes", "No"])
st.write(selected)

st.multiselect('Colors :', ["Green", "Yellow", "Red"], ["Red"])

st.select_slider('Range:', [0, 1, 2])
if st.button("Click"):
    st.write("clicked")
else:
    st.write("none")

value = st.text_input('Your type', value="Cuong")
st.write(value)
st.divider()

upload_files = st.file_uploader('Choose file', accept_multiple_files=True)
for upload_file in upload_files:
    read_file = upload_file.name
    st.write(read_file)

st.divider()
with st.form("My form"):
    col1, col2 = st.columns(2)
    col1.write("Name:")
    form_name = col1.text_input("Your name")
    col2.write("Age:")
    form_age = col2.text_input("Your Age") 

    submit = st.form_submit_button("submit")
    if submit:
        st.write(f"Name: {form_name}, Age: {form_age}")

st.divider()

import random
value = random.randint(0, 1)
if 'key' not in st.session_state:
    st.session_state["email"] = value
    st.session_state["password"] = value
st.write(st.session_state.email, st.session_state.password)
