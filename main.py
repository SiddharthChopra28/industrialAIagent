import os
import sys
import threading
import socket
import json
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

should_run_socket_listener = True
socket_listener_status = "Not Started"
PORT = 65432
CSV_FILENAME = "data_averages.csv"
CHUNK_SIZE = 10 

def generate_unique_batch_id(requested_id=None):
    existing_ids = set()
    if os.path.exists(CSV_FILENAME):
        try:
            with open(CSV_FILENAME, mode='r') as file:
                reader = csv.reader(file)
                _ = next(reader, None)
                for row in reader:
                    if row:
                        existing_ids.add(row[0])
        except Exception:
            existing_ids = set()

    if requested_id and requested_id not in existing_ids:
        return requested_id
    if requested_id:
        return f"{requested_id}_{time.time_ns()}"
    return f"batch_{time.time_ns()}"

def socket_listener_thread():
    global socket_listener_status
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server_socket.bind(('', PORT))
        server_socket.listen()
        # Non-blocking or short timeout to check should_run_socket_listener periodically
        server_socket.settimeout(1.0)
        
        socket_listener_status = f"Listening on port {PORT}"
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Batch ID", "cold_water_flow_avg", "inlet_temp_avg", "fouling_factor_avg", "ambient_temp_avg", "outlet_temp_avg", "Timestamp_avg", "Readings_Count", "Used_For_Training"])

        while should_run_socket_listener:
            try:
                conn, addr = server_socket.accept()
                socket_listener_status = f"Connected by {addr}"
                
                with conn:
                    # Set timeout for receive operations
                    conn.settimeout(1.0)
                    buffer = ""
                    current_batch_id = None
                    batch_data = {"cold_water_flow": 0, "inlet_temp": 0, "fouling_factor": 0, "ambient_temp": 0, "outlet_temp": 0, "timestamp": 0}
                    batch_count = 0
                    
                    while should_run_socket_listener:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                socket_listener_status = f"Listening on port {PORT}"
                                break
                                
                            buffer += data.decode('utf-8')
                            
                            # Process complete JSON objects
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                if not line.strip():
                                    continue
                                
                                try:
                                    message = json.loads(line)
                                    
                                    # Check for special command to start a new batch
                                    if message.get("command") == "NEW_BATCH":
                                        current_batch_id = generate_unique_batch_id(message.get("batch_id"))
                                        batch_data = {"cold_water_flow": 0, "inlet_temp": 0, "fouling_factor": 0, "ambient_temp": 0, "outlet_temp": 0, "timestamp": 0}
                                        batch_count = 0
                                        # Create new entry in CSV
                                        update_csv(current_batch_id, batch_data, batch_count)
                                        continue
                                        
                                    if current_batch_id is None:
                                        # Data received without batch, start default batch
                                        current_batch_id = generate_unique_batch_id()
                                        
                                    if "cold_water_flow" in message and "inlet_temp" in message and "fouling_factor" in message and "ambient_temp" in message and "outlet_temp" in message:
                                        # Accumulate data
                                        batch_data["cold_water_flow"] += message["cold_water_flow"]
                                        batch_data["inlet_temp"] += message["inlet_temp"]
                                        batch_data["fouling_factor"] += message["fouling_factor"]
                                        batch_data["ambient_temp"] += message["ambient_temp"]
                                        batch_data["outlet_temp"] += message["outlet_temp"]
                                        batch_data["timestamp"] += message.get("timestamp", time.time())
                                        batch_count += 1
                                        
                                        # If chunk size reached, calculate average and update CSV
                                        if batch_count % CHUNK_SIZE == 0:
                                            update_csv(current_batch_id, batch_data, batch_count)
                                            
                                except json.JSONDecodeError:
                                    pass # Ignore invalid JSON
                                    
                        except socket.timeout:
                            pass
                            
            except socket.timeout:
                pass
                
    except Exception as e:
        socket_listener_status = f"Error: {e}"
    finally:
        server_socket.close()

def update_csv(batch_id, batch_data, batch_count):
    if batch_count == 0:
        cold_water_flow_avg = inlet_temp_avg = fouling_factor_avg = ambient_temp_avg = outlet_temp_avg = ts_avg = 0
    else:
        cold_water_flow_avg = batch_data["cold_water_flow"] / batch_count
        inlet_temp_avg = batch_data["inlet_temp"] / batch_count
        fouling_factor_avg = batch_data["fouling_factor"] / batch_count
        ambient_temp_avg = batch_data["ambient_temp"] / batch_count
        outlet_temp_avg = batch_data["outlet_temp"] / batch_count
        ts_avg = batch_data["timestamp"] / batch_count
        
    rows = []
    updated = False
    
    # Read existing data
    if os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='r') as file:
            reader = csv.reader(file)
            headers = next(reader, None)
            if headers:
                rows.append(headers)
            for row in reader:
                if len(row) > 0 and row[0] == batch_id:
                    # Update existing batch entry
                    rows.append([batch_id, cold_water_flow_avg, inlet_temp_avg, fouling_factor_avg, ambient_temp_avg, outlet_temp_avg, ts_avg, batch_count, "False"])
                    updated = True
                else:
                    # Keep existing rows intact, add padding if missing field
                    if len(row) == 8:
                        row.append("False")
                    rows.append(row)
                    
    # If batch not found, append it
    if not updated:
        rows.append([batch_id, cold_water_flow_avg, inlet_temp_avg, fouling_factor_avg, ambient_temp_avg, outlet_temp_avg, ts_avg, batch_count, "False"])
        
    # Write back to file
    with open(CSV_FILENAME, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Please add it to your .env file.")
        sys.exit(1)

    # Initialize the Groq model
    # Example models: "llama-3.1-8b-instant", "llama-3.1-70b-versatile"
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.7
    )

    # Bind tools to the LLM
    from tools import extract_key_variables, fetch_batch_by_timestamp
    llm_with_tools = llm.bind_tools([extract_key_variables, fetch_batch_by_timestamp])

    system_prompt_text = (
        "You are a highly analytical AI assistant specializing in data analysis and model prediction. "
        "Your primary task is to reply to user queries by leveraging your available tools.\n\n"
        "You have access to two tools:\n"
        "1. `extract_key_variables`: Uses an XGBoost model updated with the latest data to provide Pearson correlation "
        "coefficients of SHAP values for variables Cold Water Flow Rate, Inlet Temperature, Fouling Factor, and Ambient Air Temperature. It also predicts the Outlet Temperature if you provide values for these variables.\n"
        "2. `fetch_batch_by_timestamp`: Fetches the precise data batch from the log that is closest to a provided timestamp.\n\n"
        "Instructions for your tasks:\n"
        "- PREDICTIONS: If a user asks for a prediction (providing values for Cold Water Flow Rate, Inlet Temperature, Fouling Factor, Ambient Air Temperature), call `extract_key_variables` "
        "passing those values. Present the predicted Outlet Temperature to the user AND explain how each variable correlates with "
        "the output. Use the SHAP correlations ('feature_correlations') to explain which variables had the most impact.\n"
        "- BATCH DETAILS: If a user asks for details about a particular batch entry (including what went wrong) and provides a timestamp, "
        "you MUST call BOTH tools in the same response: (1) `fetch_batch_by_timestamp` to retrieve the batch data, and (2) `extract_key_variables` "
        "to retrieve the Pearson correlations of the SHAP values. Then use the retrieved batch values and the feature weights to justify the readings."
        "- OUTPUT FORMAT: ALWAYS STRICTLY RETURN ONLY THE TEXT OUTPUT OF THE RESPONSE. DO NOT RETURN THE WHOLE JSON."

    )

    # Initialize chat history with a system message
    chat_history = [
        SystemMessage(content=system_prompt_text)
    ]

    # Start background socket listener thread
    global should_run_socket_listener
    listener_thread = threading.Thread(target=socket_listener_thread, daemon=True)
    listener_thread.start()

    print("Welcome to the Industrial AI CLI Agent!")
    print(f"Socket listener connection string: localhost:{PORT}")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            # Print socket status above input prompt
            sys.stdout.write(f"\033[90m[Background Socket]: {socket_listener_status}\033[0m\n")
            
            # Get user input
            user_input = input("\033[92mYou:\033[0m ")
            
            # Check for exit commands
            if user_input.lower().strip() in ['exit', 'quit']:
                print("\033[93mAgent:\033[0m Goodbye! Have a great day!")
                should_run_socket_listener = False
                break
                
            if not user_input.strip():
                continue

            # Append the user's message to the chat history
            chat_history.append(HumanMessage(content=user_input))

            # Fetch the AI's response
            response = llm_with_tools.invoke(chat_history)
            chat_history.append(response)

            if response.tool_calls:
                from langchain_core.messages import ToolMessage
                import json
                
                for tool_call in response.tool_calls:
                    print(f"\n\033[94m[Agent is calling tool: {tool_call['name']}]\033[0m")
                    if tool_call["name"] == "extract_key_variables":
                        result = extract_key_variables.invoke(tool_call)
                    elif tool_call["name"] == "fetch_batch_by_timestamp":
                        result = fetch_batch_by_timestamp.invoke(tool_call)
                    else:
                        result = {"error": f"Unknown tool: {tool_call['name']}"}
                        
                    # Add tool response back to the conversational history
                    if isinstance(result, ToolMessage):
                        chat_history.append(result)
                    else:
                        # Handle potential numpy types if we get raw dict
                        def default_serializer(obj):
                            if type(obj).__module__ == 'numpy':
                                return obj.item()
                            raise TypeError
                        chat_history.append(ToolMessage(
                            content=json.dumps(result, default=default_serializer), 
                            tool_call_id=tool_call["id"]
                        ))
                
                # Fetch AI response after tool execution to provide the final answer
                response = llm_with_tools.invoke(chat_history)
                chat_history.append(response)

            # if (response.content[0]["type"] == "text"):
            #     print(f"\033[93mAgent:\033[0m {response.content[0]["text"]}\n")
            # else:
            print(f"\033[93mAgent:\033[0m {response.content}\n")

        except KeyboardInterrupt:
            print("\n\033[93mAgent:\033[0m Session interrupted. Goodbye!")
            should_run_socket_listener = False
            break
        except Exception as e:
            print(f"\n\033[91mAn error occurred:\033[0m {e}\n")

    # Wait briefly for thread to detect shutdown
    if listener_thread.is_alive():
        listener_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()
