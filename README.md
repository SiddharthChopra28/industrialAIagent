# Industrial AI agent
### About & Use case
An AI agent that is capable of ingesting continuous data from a real process, and using tools to analyze this data to provide inference about the process to the user.  
It understands the correlations between and the impact of each parameter/variable on the process output, and is able to use this context to answer various types of queries from the user.  
This is useful in almost all industrual processes. Using this tool, any engineer can understand the dynamics of the process, and even predict output values for different process variables.
This works on the principle of Explainable AI - not just giving a prediction but explaining the thought process and correlations behind it.

### Technical details
- The agent currently has access to 2 tools, one to get details about any particular batch from the database, and the other to interface with the ML model.
- The ML model is a basic eXtreme Gradient Boost (XGBoost) model, which is used for prediction of outputs based on process conditions. SHAP analysis on this model tells us about the correlations b/w and the effects of these parameters.
- The model undergoes continuous training; a base trained model is stored, and whenever the tool is called, the model is tuned on the newly received batch data from the db.
- The agent is conversational and maintains a complete chat history, and context for the LLM. The LLM is currently the gpt-oss-120b model from groq.
- Currently the agent is in the form of a CLI application, which may be later hosted on the cloud and presented as a chat interface on the web.
- The live data ingestion currently is a simulator that works on fabricated data, but can later be connected to any industrial sensors using microcontrollers.
