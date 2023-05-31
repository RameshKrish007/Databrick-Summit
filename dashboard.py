import cv2
import streamlit as st
import base64
import numpy as np
import pandas as pd
import sys
import openai
from streamlit_chat import message
import requests
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

openai.api_key = st.secrets["api_secret"]

#Header backend start
import openai
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
import streamlit as st

import json
from databricks_cli.sdk.api_client import ApiClient
from databricks_api import DatabricksAPI
from databricks_cli.clusters.api import ClusterApi
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.cluster_policies.api import ClusterPolicyApi
from databricks_cli.dbfs.dbfs_path import DbfsPath
from databricks_cli.dbfs.api import DbfsApi
from dotenv import load_dotenv
load_dotenv()
#Header End for backend
# pip install streamlit-chat  
from streamlit_chat import message
def main(username="User"):
    activities = ["Cluster", "Policy", "DPFS"]
    st.sidebar.image("assets\\images\\LTIlogo.png")
    choice = st.sidebar.selectbox("Easy Access", activities)
    if choice == "Cluster":
        st.sidebar.markdown("###")
        st.sidebar.markdown("###")
        st.sidebar.markdown("###")
        st.sidebar.markdown("<h3 class='mb-0'>About Cluster</h3>", unsafe_allow_html=True)
        st.sidebar.markdown("In Databricks cluster perform operations like able to view list of cluster, create, Restart, Terminate Cluster etc...")
        st.title("DataBricks Operation")
        api_client = ApiClient(
            host="https://lti-datascience-coe.cloud.databricks.com",
            token="dapif02529dfb5d9e122a125a91d125073d7", verify=False
            )

        db = DatabricksAPI(
            host="https://lti-datascience-coe.cloud.databricks.com",
            token="dapif02529dfb5d9e122a125a91d125073d7", verify=False
        )

        clusters_api = ClusterApi(api_client)
        cluster_policies_api = ClusterPolicyApi(api_client)
        dbfs_api = DbfsApi(api_client)


        API_KEY = "18bc8a0873114be0be0d32d6e05122d4"

        openai.api_type = "azure"
        openai.api_key = API_KEY
        openai.api_base = "https://coe-azopenai-scus.openai.azure.com/" 
        openai.api_version = "2022-12-01"

        url = openai.api_base + "/openai/deployments?api-version=2022-12-01"

        r = requests.get(url, headers={"api-key": API_KEY})


        def normalize_text(s, sep_token = " \n "):
            s = re.sub(r'\s+',  ' ', s).strip()
            s = re.sub(r". ,","",s)
            # remove all instances of multiple spaces
            s = s.replace("..",".")
            s = s.replace(". .",".")
            s = s.replace("\n", "")
            s = s.strip()
            
            return s

        def search_docs(df, user_query, top_n=3, to_print=False):
            embedding = get_embedding(
                user_query,
                engine="text-search-curie-query-001"
            )
            df["similarities"] = df.curie_search.apply(lambda x: cosine_similarity(x, embedding))

            res = (
                df.sort_values("similarities", ascending=False)
                .head(top_n)
            )
            if to_print:
                display(res)
            return res

        def cip(query,p):
            prompt=f"""Identify the parameters ({p}) \
        from the following context. Format your answer as a single value corresponding to the parameter and set as NA if the value is not found within the context.: \n
        Context: \n{query}\n
        Parameters:\n({p})=(
        """
            #print(prompt)

            out = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            suffix=")")
            s=out['choices'][0]['text']
            return True,s

        def check_in_prompt(query, param):
            params={}
            if res.iloc[0]['Parameters']!='NOPARAM':
                for p in res.iloc[0]['Parameters'].split(','):
                    status, param = cip(query,p)
                    if status == True:
                        params[p]=normalize_text(param)
            return True,params





        ##########################################
        #Clusters
        ##########################################

        def clusters_list(param):
            ls = []
            for cl in clusters_api.list_clusters()['clusters']:
                cl_id = cl['cluster_id']
                cl_name = cl['cluster_name']
                cl_state = cl['state']
                ls.append([cl_id,cl_name,cl_state])
            df=pd.DataFrame(ls,columns=['cluster_id','cl_name','cl_state']) 
            return df

        def get_cluster(cluster_id):
            cluster = db.cluster.get_cluster(cluster_id)
            # Extract relevant information from the cluster details
            cluster_info = {
                "cluster_id": cluster["cluster_id"],
                "cluster_name": cluster["cluster_name"],
                "spark_version": cluster["spark_version"],
                "node_type_id": cluster["node_type_id"],
                #"num_workers": cluster["num_workers"],
                "State" : cluster["state"]
            }
            # Convert the extracted information into a DataFrame
            df = pd.DataFrame([cluster_info])
            return df

        def terminate_clusters(cluster_id):
            clusters_api.delete_cluster(cluster_id)
            
        def delete_clusters(cluster_id):
            clusters_api.permanent_delete(cluster_id)
            
        def start_clusters(cluster_id):
            clusters_api.start_cluster(cluster_id)
            
        def restart_clusters(cluster_id):
            clusters_api.restart_cluster(cluster_id)

        def create_cluster(cluster_name, spark_version, node_type_id,num_workers):
            try:
                cluster = db.cluster.create_cluster(
                    cluster_name=cluster_name,
                    spark_version=spark_version,
                    node_type_id=node_type_id,
                    num_workers=num_workers
                )
                return cluster
            except Exception as e:
                print('Error creating Databricks cluster:', str(e))
                return None

        def edit_cluster(cluster_id,num_workers,cluster_name,spark_version,node_type_id):
            db.cluster.edit_cluster(cluster_id = cluster_id,
                                    num_workers = num_workers,
                                    cluster_name = cluster_name,
                                    spark_version = spark_version,
                                    node_type_id = node_type_id)
            
        def resize_cluster(cluster_id,num_workers):
            db.cluster.resize_cluster(
                cluster_id = cluster_id ,
                num_workers= num_workers)

        ##########################################
        #Cluster Policy
        #########################################
            
        def cluster_policy_list(param):
            ls = []
            for pl in cluster_policies_api.list_cluster_policies()['policies']:
                pl_id = pl['policy_id']
                pl_name = pl['name']
                ls.append([pl_id,pl_name])
            df= pd.DataFrame(ls, columns=['policy_id','name'])
            st.dataframe(df)
            return df
            
        # def get_cluster_policy(policy_id):
        #     cluster_policies_api.cluster_policies.get_policy(policy_id)

        def get_cluster_policy(policy_id):
            return db.policy.get_policy(policy_id)

        def create_cluster_policy(policy_name, config_file_path):
            api_endpoint = "https://lti-datascience-coe.cloud.databricks.com/api/2.0/policies/clusters/create"  # Replace <databricks-instance> with your Databricks instance URL
            api_token = "dapi84ec6d8a505f07d68284d5e84c7649ac"
            headers = {
                "Authorization": "Bearer " + api_token,
                "Content-Type": "application/json"
            }

            with open(config_file_path, 'r') as file:
                config_data = json.load(file)

            config_data["name"] = policy_name

            response = requests.post(api_endpoint, headers=headers, json=config_data)
            response_json = response.json()

            if response.status_code == 200:
                print("Cluster policy created successfully with ID:", response_json["policy_id"])
            else:
                print("Error creating cluster policy:", response_json["message"])
                
        def edit_cluster_policy(policy_id,policy_name,definition):
            return cluster_policies_api.edit_policy(policy_id,policy_name,definition)

        def delete_clusters_ploicy(policy_id):
            return cluster_policies_api.delete_cluster_policy(policy_id)


        #############################################
        # DBFS
        ################################################
        def list_files(path):
            return db.dbfs.list(path=path)
            
        def make_dirs(path):
            db.dbfs.mkdirs(path=path)
            
        def move_file(source_path,destination_path):
            db.dbfs.move(source_path = source_path,
                        destination_path = destination_path
                        )
        def get_status(path):
            return db.dbfs.get_status(path=path)

        def file_exists(path):
            dbfs_path = DbfsPath(path)
            return dbfs_api.file_exists(dbfs_path)

        def copy_files(source_path,destination_path):
            dbfs_api.cp(recursive=None,
                        overwrite=None,
                        src=source_path, 
                        dst=destination_path,
                        headers=None)
            


        def list_databricks_cluster(param):
            st.dataframe(clusters_list(''))

        def get_databricks_cluster(dictionary):
            print('Dict:',dictionary)
            pids=[str(c) for c in clusters_list('')['cluster_id']]
            status = True
            running=True
            #while running:
            if dictionary['cluster_id'] == 'NA':
                st.session_state.generated.append("There is no cluster_id present in the query, so please enter the cluster_id from the below list")
                list_databricks_cluster('')
                inp = placeholder.text_input(f"Please enter the correct cluster_id or enter'quit' to exit\n",key="getc1",on_change=submit)
                if inp=='quit':
                    running=False
                elif inp in pids:
                    out=get_cluster(inp)
                    st.session_state.generated.append("Successfully got the cluster:")
                    st.dataframe(out)
                    running=False
                else:
                    dictionary['cluster_id']=inp
            elif dictionary['cluster_id'] != 'NA' and dictionary['cluster_id'] in pids:
                out=get_cluster(dictionary['cluster_id'])
                st.session_state.generated.append.write("Successfully got the cluster: ",out)
                st.dataframe(out)
                running=False
            else:
                st.session_state.generated.append.write("The cluster_id provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                list_databricks_cluster('')
                inp=placeholder.text_input(f"Please enter the correct cluster_id or enter'quit' to exit\n",key="getc2",on_change=submit)
                if inp=='quit':
                    running=False
                elif inp in pids:
                    out=get_cluster(inp)
                    st.session_state.generated.append.write("Successfully got the cluster:")
                    st.dataframe(out)
                    running=False
                else:
                    dictionary['cluster_id']=inp


        def create_databricks_clusters(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("Create cluster started.....")
                    print("Please give these inputs to create cluster....")
                    cluster_name = input("Enter the cluster name: ")
                    spark_version = input("Enter the Spark version: ")
                    node_type_id = input("Enter the node type ID: ")
                    num_workers = input("Enter the num_workers: ")
                    response= create_cluster(cluster_name, spark_version, node_type_id,num_workers)
                    print("Cluster Created Successfully....",response["cluster_id"])
                    inp=input("Cluster is created,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to Create the cluster please Check....")
                    else:
                        return None



        def edit_databricks_cluster(dictionary):
            print('Dict:',dictionary)
            cids=[str(c) for c in clusters_list('')['cluster_id']]
            status = True
            running=True
            while running:
                print("please enter the cluster_id from the below list")
                print(clusters_list(''))
                inp=input("Please enter the correct cluster_id or enter'quit' to exit\n")
                if inp=='quit':
                    running=False
                elif inp in cids:
                    current_cluster=get_cluster(inp)
                    print(current_cluster)
                    cluster_id = current_cluster.iloc[0]['cluster_id']
                    num_workers = input('Please enter the number of workers....')
                    cluster_name = input('Please enter the name of cluster_name...')
                    spark_version = input('Please enter the new spark Version....')
                    node_type_id = input('Please enter the new node_type_id...')
                    db.cluster.edit_cluster(cluster_id = cluster_id,
                                            num_workers = num_workers,
                                            cluster_name = cluster_name,
                                            spark_version = spark_version,
                                            node_type_id = node_type_id)
                    updated_cluster = get_cluster(cluster_id)
                    result_df = pd.concat([current_cluster, updated_cluster], keys=['Before', 'After'])
                    print(result_df)
                    running=False
                else:
                    print("The cluster_id provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    inp=input("Please enter the correct cluster_id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    else:
                        None



        def resize_databricks_cluster(dictionary):
            print('Dict:',dictionary)
            cids=[str(c) for c in clusters_list('')['cluster_id']]
            status = True
            running=True
            while running:
                print("please enter the cluster_id from the below list")
                print(clusters_list(''))
                inp=input("Please enter the correct cluster_id or enter'quit' to exit\n")
                if inp=='quit':
                    running=False
                elif inp in cids:
                    current_cluster=get_cluster(inp)
                    print(current_cluster)
                    cluster_id = current_cluster.iloc[0]['cluster_id']
                    num_workers = input('Please enter the number of workers....')
                    resize_cluster(cluster_id = cluster_id,num_workers = num_workers)
                    updated_cluster = get_cluster(cluster_id)
                    result_df = pd.concat([current_cluster, updated_cluster], keys=['Before', 'After'])
                    print(result_df)
                    running=False
                else:
                    print("The cluster_id provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    inp=input("Please enter the correct cluster_id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    else:
                        None


        def start_databricks_cluster(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                if 'cluster_id' not in dictionary:
                    print("There is no cluster_id present in the query, so please enter the cluster_id from the below list")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        start_clusters(cluster_id)
                        print("Successfully Started the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp
                elif 'cluster_id' in dictionary and dictionary['cluster_id'] in [str(cid) for cid in clusters_list('')['cluster_id']]:
                    start_clusters(dictionary['cluster_id'])
                    print("Successfully Started the clusters:",dictionary['cluster_id'])
                    running=False
                else:
                    print("The clusterid provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        start_clusters(cluster_id)
                        print("Successfully Started the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp


        def restart_databricks_cluster(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                if 'cluster_id' not in dictionary:
                    print("There is no cluster_id present in the query, so please enter the cluster_id from the below list")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        restart_clusters(cluster_id)
                        print("Successfully Restarted the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp
                elif 'cluster_id' in dictionary and dictionary['cluster_id'] in [str(cid) for cid in clusters_list('')['cluster_id']]:
                    restart_clusters(dictionary['cluster_id'])
                    print("Successfully Restarted the clusters:",dictionary['cluster_id'])
                    running=False
                else:
                    print("The clusterid provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        restart_clusters(cluster_id)
                        print("Successfully Restarted the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp


        def terminate_cluster_from_cluster_list(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                if 'cluster_id' not in dictionary:
                    print("There is no cluster_id present in the query, so please enter the cluster_id from the below list")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        terminate_clusters(cluster_id)
                        print("Successfully Terminated the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp
                elif 'cluster_id' in dictionary and dictionary['cluster_id'] in [str(cid) for cid in clusters_list('')['cluster_id']]:
                    terminate_clusters(dictionary['cluster_id'])
                    print("Successfully Terminated the clusters:",dictionary['cluster_id'])
                    running=False
                else:
                    print("The clusterid provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        terminate_clusters(cluster_id)
                        print("Successfully Terminated the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp


        def permanent_delete_cluster_from_list(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                if 'cluster_id' not in dictionary:
                    print("There is no cluster_id present in the query, so please enter the cluster_id from the below list")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        delete_clusters(cluster_id)
                        print("Successfully Deleted Permanently the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp
                elif 'cluster_id' in dictionary and dictionary['cluster_id'] in [str(cid) for cid in clusters_list('')['cluster_id']]:
                    delete_clusters(dictionary['cluster_id'])
                    print("Successfully Deleted Permanently the clusters:",dictionary['cluster_id'])
                    running=False
                else:
                    print("The clusterid provided does not exist so please enter the cluster_id from below list or enter 'quit'")
                    print(clusters_list(''))
                    inp=input("Please enter the correct cluster id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in clusters_list('')['cluster_id']:
                        delete_clusters(cluster_id)
                        print("Successfully Deleted Permanently the clusters:",inp)
                        running=False
                    else:
                        dictionary['cluster_id']=inp


        # ### Databricks Policy


        def get_databricks_policy(dictionary):
            print('Dict:',dictionary)
            pids=[str(c) for c in cluster_policy_list('')['policy_id']]
            status = True
            running=True
            while running:
                if 'policy_id' not in dictionary:
                    print("There is no policy_id present in the query, so please enter the policy_id from the below list")
                    print(cluster_policy_list(''))
                    inp=input("Please enter the correct policy id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in pids:
                        out=get_cluster_policy(inp)
                        df = pd.DataFrame([{
                                'Policy ID': out['policy_id'],
                                'Policy Name': out['name'],
                                'Definition': json.dumps(out['definition']),
                                'created_at_timestamp' : out['created_at_timestamp'],
                                'is_default' : out['is_default']
                                }])
                        print("Successfully get the policy:",df)
                        running=False
                    else:
                        dictionary['policy_id']=inp
                elif 'policy_id' in dictionary and dictionary['policy_id'] in pids:
                    out=get_cluster_policy(dictionary['policy_id'])
                    df = pd.DataFrame([{
                                'Policy ID': out['policy_id'],
                                'Policy Name': out['name'],
                                'Definition': json.dumps(out['definition']),
                                'created_at_timestamp' : out['created_at_timestamp'],
                                'is_default' : out['is_default']
                                }])
                    print("Successfully get the policy: ",df)
                    running=False
                else:
                    print("The policyid provided does not exist so please enter the policy_id from below list or enter 'quit'")
                    print(cluster_policy_list(''))
                    inp=input("Please enter the correct policy id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in pids:
                        out=get_cluster_policy(inp)
                        df = pd.DataFrame([{
                                'Policy ID': out['policy_id'],
                                'Policy Name': out['name'],
                                'Definition': json.dumps(out['definition']),
                                'created_at_timestamp' : out['created_at_timestamp'],
                                'is_default' : out['is_default']
                                }])
                        print("Successfully get the policy: ",df)
                        running=False
                    else:
                        dictionary['policy_id']=inp


        def create_databricks_cluster_policies(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("Create cluster policy started.....")
                    print("Please give these inputs to create cluster policy....")
                    policy_name = input("Enter the cluster name: ")
                    config_file_path = input("Enter the JSON config_file_path file: ")
                    create_cluster_policy(policy_name, config_file_path)
                    print("Cluster policy Created Successfully....")
                    inp=input("Cluster policy is created,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to Create the cluster policy please Check....")
                    else:
                        return None


        def edit_databricks_cluster_policy(dictionary):
            print('Dict:',dictionary)
            pids=[str(c) for c in cluster_policy_list('')['policy_id']]
            status = True
            running=True
            while running:
                print("please enter the policy_id from the below list")
                print(cluster_policy_list(''))
                inp=input("Please enter the correct policy_id or enter'quit' to exit\n")
                if inp=='quit':
                    running=False
                elif inp in pids:
                    current_policy=get_cluster_policy(inp)
                    print(current_policy)
                    policy_id = get_cluster_policy['policy_id']
                    policy_name = input('Please enter the policy_name....')
                    definition = input('Please enter the definition as JSON file...')
                    edit_cluster_policy(policy_id = policy_id,
                                        policy_name = policy_name,
                                        definition = definition)
                    updated_policy = get_cluster_policy['policy_id']
                    result_df = pd.concat([current_policy, updated_policy], keys=['Before', 'After'])
                    print(result_df)
                    running=False
                else:
                    print("The policy_id provided does not exist so please enter the policy_id from below list or enter 'quit'")
                    inp=input("Please enter the correct policy_id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    else:
                        None


        def delete_policy_from_policy_list(dictionary):
            print('Dict:',dictionary)
            pids=[str(c) for c in cluster_policy_list('')['policy_id']]
            status = True
            running=True
            while running:
                if 'policy_id' not in dictionary:
                    print("There is no policy_id present in the query, so please enter the policy_id from the below list")
                    print(cluster_policy_list(''))
                    inp=input("Please enter the correct policy id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in pids:
                        delete_clusters_ploicy(inp)
                        print("Successfully Deleted the policy:",inp)
                        running=False
                    else:
                        dictionary['policy_id']=inp
                elif 'policy_id' in dictionary and dictionary['policy_id'] in pids:
                    delete_clusters_ploicy(dictionary['policy_id'])
                    print("Successfully Deleted the policy:",inp)
                    running=False
                else:
                    print("The policyid provided does not exist so please enter the policy_id from below list or enter 'quit'")
                    print(cluster_policy_list(''))
                    inp=input("Please enter the correct policy id or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                    elif inp in pids:
                        delete_clusters_ploicy(inp)
                        print("Successfully Deleted the policy:",inp)
                        running=False
                    else:
                        dictionary['policy_id']=inp

        def list_databricks_file(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("Listing files databricks started.....")
                    print("Please give the path as input to list the files....")
                    path = input("Enter the databricks path: ")
                    files = list_files(path=path)
                    df = pd.DataFrame(files)
                    print("The files in the file list", df)
                    print("listing the files in Databricks successful....")
                    inp=input("file listing has successfull done,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to list the files in Databricks please Check....")
                    else:
                        return None



        def databricks_file_status(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("databricks file status started.....")
                    print("Please give the path as input to check the status....")
                    path = input("Enter the databricks path: ")
                    file_status=get_status(path)
                    print("The file status:--",file_status)
                    print("databricks file status successful....")
                    inp=input("databricks file status check successfull done,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to check the Databricks file status please Check....")
                    else:
                        return None


        def databricks_create_directory(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("databricks directory creation started.....")
                    print("Please give the path as input to check the status....")
                    path = input("Enter the databricks path: ")
                    make_dirs(path)
                    print("databricks directory created successful....")
                    inp=input("databricks directory creation successfully done,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to create Databricks directory ,please Check....")
                    else:
                        return None

        def databricks_file_move(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("databricks file moving started.....")
                    print("Please give the path as input to check the status....")
                    source_path = input("Enter the source databricks path")
                    destination_path = input("Enter the destination databricks path: ")
                    move_file(source_path,destination_path)
                    print("Moving file from source to destination in Databricks has done successfully....")
                    inp=input("Moving file from source to destination in Databricks has done successfully,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to move file from source to destination in Databricks ,please Check....")
                    else:
                        return None



        def databricks_file_exists(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("databricks checking file exists started.....")
                    print("Please give the path as input to check the status....")
                    path = input("Enter the source databricks path ")
                    file_check = file_exists(path)
                    print("Checking file in Databricks has done successfully....", file_check)
                    inp=input("Checking file in Databricks has done successfully,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to check files in Databricks ,please Check....")
                    else:
                        return None


        def databricks_file_copy(dictionary):
            print('Dict:',dictionary)
            status = True
            running=True
            while running:
                try:
                    print("databricks copying file started.....")
                    print("Please give the path as input to check the status....")
                    source_path = input("Enter the source databricks path")
                    destination_path = input("Enter the destination databricks path: ")
                    copy_files(recursive=None,
                            overwrite=None,
                            src=source_path,
                            dst=destination_path,
                            headers=None
                        )
                    print("Copying file in Databricks has done successfully....")
                    inp=input("copying file in Databricks has done successfully,Please enter'quit' to exit\n")
                    if inp=='quit':
                        running=False 
                except Exception as e:
                    inp=input("Please enter the correct input or enter'quit' to exit\n")
                    if inp=='quit':
                        running=False
                        print("Failed to copy files in Databricks ,please Check....", str(e))
                    else:
                        return None


        #One session i started the state is set to start
        func_dict={'list_clusters()':list_databricks_cluster,
                'get_cluster()':get_databricks_cluster,
                'create_cluster()' : create_databricks_clusters,
                'edit_cluster()' : edit_databricks_cluster,
                'resize_cluster()' :resize_databricks_cluster,
                'start_cluster()':start_databricks_cluster,
                'restart_cluster()' :restart_databricks_cluster,
                'terminate_cluster()':terminate_cluster_from_cluster_list,
                'delete_cluster()':permanent_delete_cluster_from_list,
                'list_cluster_policies()':cluster_policy_list,
                'get_cluster_policy()': get_databricks_policy,
                'create_cluster_policy()': create_databricks_cluster_policies,
                'edit_cluster_policy()' : edit_databricks_cluster_policy,
                'delete_cluster_policy()':delete_policy_from_policy_list,
                'list_files()' : list_databricks_file,
                'get_status()' : databricks_file_status,
                'mkdirs()' : databricks_create_directory,
                'move()' : databricks_file_move,
                'file_exists()' : databricks_file_exists,
                'copy()': databricks_file_copy}

        df = pd.read_csv('Databricks_api_with_embedings_updated.csv',encoding = 'utf8',usecols=['API','Description','Parameters','token_count','curie_search'])
        df["curie_search"] = df.curie_search.apply(eval).apply(np.array)
        df_cluster = df[0:9]
        df_policy = df[9:14]
        df_dbfs = df[14:]


        ### Kasi and Ramesh



        #def get_text(placeholder,key):
        #    st.text_input("",placeholder=placeholder, key=key,on_change=submit)
        #    return st.session_state.something

        def submit():
            placeholder.empty()
            
        if 'something' not in st.session_state:
            st.session_state.something = ''
            
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

        if "disabled" not in st.session_state:
            st.session_state["disabled"] = False

        def disable(inp):
            st.session_state["disabled"] = inp

        placeholder = st.empty()

        state='start'
        #sel = input("Enter your selection: cluster,plicy,dbfs:\n")
        #selection = ['cluster','policies','dbfs']
        #if sel == "cluster":
        #    df = df_cluster
        #elif sel == "policy":
        #    df=df_policy
        #elif sel == "dbfs":
        #    df=df_dbfs

        #user_input = get_text()
        count=1
        #while state != 'stop':
        #placeholder = st.empty()
        
        query = placeholder.text_input("",placeholder='Welcome to Databricks Codex.Enter your query/input',key="inp1",on_change=submit)
        st.session_state.past.append('Welcome to Databricks Codex.Enter your query/input')
        #del st.session_state["inp1"+str(count)]
        count+=1


        #query=get_text("Welcome to Databricks Codex.Enter your query/input", "inp1")
        st.session_state.past.append(query)
        if query !='':
            res=search_docs(df, query, top_n=4)
            api=res.iloc[0]['API']
            _,param = check_in_prompt(query,res.iloc[0]['Parameters'])
            st.session_state.generated.append(f"Please enter (y/n) is you want to execute below API: {api}")
            inp = placeholder.text_input(f"Please enter (y/n) is you want to execute below API:\n{api}",key="inp2",on_change=submit)
            #del st.session_state["inp2"+str(count)]
            #count+=1
            
            st.session_state.past.append(inp)
            if inp=='y':
                print(func_dict[api](param))
                st.session_state.generated.append(f"Successfully executed the API: {api}")
            else:
                st.session_state.generated.append("Cannot undertsand your input.Let's begin from the begining")
            
            
        if st.session_state['generated']:
            
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                if i==0:
                    pass
                else:
                    message(st.session_state['past'][i-1], is_user=True, key=str(i) + '_user')