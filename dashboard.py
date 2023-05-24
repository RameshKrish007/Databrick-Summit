import cv2
# import os
import streamlit as st
import base64
# from st_click_detector import click_detector
import numpy as np
import pandas as pd
import sys
# from PIL import Image, ImageDraw
import openai 
from streamlit_chat import message
import requests
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
# openai.api_key = st.secrets["api_secret"]


def main(username="User"):
  with open('style.css') as f:
      st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 3
  activities = ["Cluster","Policy", "DPFS"]
  st.sidebar.image("assets\\images\\LTIlogo.png")
  choice=st.sidebar.selectbox("Easy Access",activities)

  def annotate(name):
      st.markdown("""
      <h3>Cluster Lists</h3>
      <table>
    <tr>
      <th>ClusterName</th>
      <th>Created Date</th>
      <th>Status</th>
    </tr>
    <tr>
      <td>Sample Cluster 1</td>
      <td>05/05/2023</td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Sample Cluster 2</td>
      <td>10/05/2023</td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Sample Cluster 3</td>
      <td>14/05/2023</td>
      <td>Active</td>
    </tr>
    <tr>
      <td>Sample Cluster 4</td>
      <td>01/05/2023</td>
      <td>Terminated</td>
    </tr>
    <tr>
      <td>Sample Cluster 5</td>
      <td>17/05/2023</td>
      <td>Inactive</td>
    </tr>
    <tr>
      <td>Sample Cluster 6</td>
      <td>04/04/2023</td>
      <td>Inactive</td>
    </tr>
  </table>
      """, unsafe_allow_html=True)

  def restart(na):
      st.markdown(f"""
      <div class="alert alert-success" role="alert">
          Selcted Cluster <strong> {na} </strong> is Restarted Successfully
      </div>""",
      unsafe_allow_html=True)

  def terminate(na):
      st.markdown(f"""
      <div class="alert alert-danger" role="alert">
          Selcted Cluster <strong> {na} </strong> is Terminated 
      </div>""",
      unsafe_allow_html=True)

      
  if choice =="Cluster":
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("<h3 class='mb-0'>About Cluster</h3>", unsafe_allow_html=True)
      st.sidebar.markdown("In Databricks cluster perform operations like able to view list of cluster, create, Restart, Terminate Cluster etc...")
      st.title("DataBricks Operation")
      input_text = st.text_input('Enter Some Input....')
      chat_button = st.button("Execute")
      clear_button = st.button("Clear")
      if chat_button and input_text.strip() != "":
          with st.spinner("Loading...💫"):
            data = requests.get('http://localhost:3000/clusterData').json()
            # st.write(data)
            dataFrame = pd.DataFrame(columns=data[0].keys())
            for i in range(len(data)):
              dataFrame.loc[i] = data[i].values()
          # st.write(dataFrame)
          st.dataframe(dataFrame, use_container_width=True)
          # st.write(dataFrame.empty)
          if chat_button and input_text.strip() == 'Y' and dataFrame.empty == False:
              data = requests.get('http://localhost:3000/terminateCluster').json()
              successMsg = data[0]['successMessage']
              st.markdown(f"""
              <div class="alert alert-danger" role="alert">
                  Selcted Cluster <strong> {successMsg} </strong> is Terminated Successfully
              </div>""",
              unsafe_allow_html=True)
          if chat_button and input_text.strip() == 'N' and dataFrame.empty == False:
                st.markdown(f"""
                <div class="alert alert-warning" role="alert">
                    Please Provide Input Y/N
                </div>""",
                unsafe_allow_html=True)
              # st.write("True")
          # else:
          #     st.markdown(f"""
          #     <div class="alert alert-warning" role="alert">
          #         Selcted Cluster <strong>  </strong> is Terminated Successfully
          #     </div>""",
          #     unsafe_allow_html=True)
                
      # else:
          # st.warning("Please enter something! ⚠")
      # na=st.text_input('Please select or provide input to perform operations')
      # if len(na)>0:
      #     annotate(na)
      # else:
      #     pass    
      #st.markdown(get_binary_file_downloader_html(original, 'Picture'), unsafe_allow_html=True)


  if choice =="Policy":
      st.title("DataBricks Operation")
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("<h3 class='mb-0'>About Policy</h3>", unsafe_allow_html=True)
      st.sidebar.markdown("In Databricks cluster perform operations like able to view list of cluster, create, Restart, Terminate Cluster etc...")
      # st.title("DataBricks Operation")
      input_text = st.text_input('Enter Some Input....')
      chat_button = st.button("Execute")
      clear_button = st.button("Clear")
      if chat_button and input_text.strip() != "":
          with st.spinner("Loading...💫"):
            data = requests.get('http://localhost:3000/aggregateData').json()
            st.write(data)
            AgGrid(data)
      else:
          st.warning("Please enter something! ⚠")
      # na=st.text_input('Enter ClusterName To Restart')
      # if len(na)>0:
      #     restart(na)
      # else:
      #     pass

  if choice =="DPFS":
      st.title("DataBricks Operation")
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("###")
      st.sidebar.markdown("<h3 class='mb-0'>About DPFS</h3>", unsafe_allow_html=True)
      st.sidebar.markdown("In Databricks cluster perform operations like able to view list of cluster, create, Restart, Terminate Cluster etc...")
      # st.title("DataBricks Operation")
      input_text = st.text_input('Enter Some Input....')
      chat_button = st.button("Execute")
      clear_button = st.button("Clear")
      if chat_button and input_text.strip() != "":
          with st.spinner("Loading...💫"):
            data = requests.get('http://localhost:3000/aggregateData').json()
            st.write(data)
            AgGrid(data)
      else:
          st.warning("Please enter something! ⚠")
      # na=st.text_input('Enter ClusterName To Terminate')
      # if len(na)>0:
      #     terminate(na)
      # else:
      #     pass