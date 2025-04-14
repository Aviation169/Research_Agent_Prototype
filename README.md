🤖Research Agent Prototype🤖
-
This repository contains a working prototype of a research agent that collects information from the web, performs sentiment analysis, and generates concise reports on given topics. 
The prototype includes a graphical user interface (GUI) built with tkinter to facilitate interaction.

`⚠️Note: This is a working prototype. For production use, significant refinements are required, including enhanced error handling, security improvements, performance optimizations, and extensive testing.`

💻Features💻
-
→Web Data Collection:
Searches for relevant URLs based on the research topic and fetches textual content from each URL.

→Sentiment Analysis:
Analyzes the sentiment of the fetched text using a pre-trained transformer model.

→Text Generation:
Generates a concise report with bullet points summarizing key findings from the gathered data.

→Graphical User Interface (GUI):
Provides a user-friendly interface using tkinter, including options to start/stop research and save outputs.

→Multi-threading:
Uses multi-threading to collect data in parallel, improving responsiveness and efficiency.

→Logging:
Implements logging to track application events and assist in debugging.

📁Dependencies📂
-
Ensure you have Python 3.x installed. The following libraries are required:

`requests`

`beautifulsoup4`

`googlesearch-python`

`torch`

`transformers`

`tkinter (included with Python)`

🖨️Setup Instructions🖨️
-
→Clone or Download the Repository

→Configure the Model:
The ResearchAgent class expects a pre-trained language model at the path specified by model_path.

Here you can download the model: `https://drive.google.com/drive/folders/1EUpuE5uDAIozW-Sqh6mquE7RJibQDvDf?usp=sharing`

→Run the Application:
Execute the Python file to launch the GUI: `python Test1.py`

💻Usage💻
-
→Enter Research Topic:
Type your research topic in the provided input field. The default value is "Tamilnadu".

→Start Research:
Click the Start Research button to begin data collection and analysis.

→Monitor Progress:
The interface will display status updates and progress, indicating when data is being collected and when analysis is underway.

→Save Output:
Once the research completes, click the Save Output button to save the generated report and collected source information to a text file.

🧑‍💻Code Structure🧑‍💻
-
→ResearchAgent:
Contains methods to collect data from URLs, perform sentiment analysis, and generate reports using a transformer-based text generation pipeline.

→ResearchUI:
Implements the tkinter GUI, providing interactive controls and display areas for input, progress, and output.

→Main Function:
Initializes the GUI and starts the main event loop.

❌Limitations and Future Improvements✅
-
→Error Handling:
Current error handling is minimal. Enhance it to better manage exceptions and edge cases in production.

→Performance:
Optimize multi-threading and network requests to improve responsiveness and resource usage.

→Model Path Management:
Consider dynamic model selection and better handling of model paths.

→User Interface:
The GUI design is basic and can be further refined to improve usability and aesthetics.

→Security:
Strengthen security measures, especially for handling external inputs and network interactions.

→Testing:
Implement comprehensive testing to ensure reliability, robustness, and to catch potential bugs before production deployment.

📃License📃
-
This project is provided as-is without any warranty. Use at your own risk. For production, further development and testing are strongly recommended.




