import logging
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from concurrent.futures import ThreadPoolExecutor
import threading
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

class ResearchAgent:
    def __init__(self, model_path: str = r"G:\My Drive\llama-3.2-3b-instruct"):
        self.data = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        self.gen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gen_model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.generator = pipeline(
            "text-generation", 
            model=self.gen_model, 
            tokenizer=self.gen_tokenizer, 
            device=0 if torch.cuda.is_available() else -1
        )
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("Agent initialized with sentiment analysis")

    def fetch_url(self, url: str) -> dict:
        """Fetch content from a URL and analyze sentiment."""
        result = {"url": url, "text": "", "sentiment": "N/A"}
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find_all(['p', 'h1', 'h2', 'li'])
            text = ' '.join(element.get_text() for element in content if element.get_text())[:1000]
            result["text"] = text
            sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
            result["sentiment"] = f"{sentiment['label']} ({sentiment['score']:.2f})"
            return result
        except requests.exceptions.Timeout:
            logger.error(f"Timeout accessing {url}")
            result["text"] = "Timeout occurred"
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            result["text"] = "Failed to retrieve"
        return result

    def collect_data(self, topic: str, stop_event: threading.Event) -> None:
        """Collect data from multiple URLs in parallel."""
        logger.info(f"Starting data collection for: {topic}")
        self.data = []
        try:
            urls = list(search(f"{topic} information recent results", num_results=5))
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.fetch_url, url) for url in urls]
                for future in futures:
                    if stop_event.is_set():
                        logger.info("Data collection stopped by user")
                        return
                    result = future.result()
                    if result["text"] and result["text"] != "Failed to retrieve":
                        self.data.append(result)
                        logger.info(f"Collected from: {result['url']}")
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
        logger.info(f"Collected {len(self.data)} items")

    def analyze_and_generate(self, topic: str) -> tuple:
        """Generate a report and analyze its sentiment."""
        logger.info(f"Analyzing and generating text for: {topic}")
        if not self.data:
            return f"No data collected for '{topic}'. Please collect data first.", "N/A"
        
        context = "\n".join([item["text"] for item in self.data])
        prompt = (
            f"Provide a concise report on {topic} based on this data. "
            f"Use bullet points for key findings and give conclusion(Your final opinion):\n{context}"
        )
        response = self.generator(
            prompt, 
            max_new_tokens=2048, 
            num_return_sequences=1, 
            truncation=True
        )[0]["generated_text"]
        response = response.replace(prompt, "").strip()
        sentiment = self.sentiment_analyzer(response[:512])[0]
        logger.info(f"Generated text for '{topic}'")
        return response, f"{sentiment['label']} ({sentiment['score']:.2f})"

class ResearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Agent")
        self.root.geometry("1000x750")
        self.root.configure(bg="#F5F6F5")

        # Modern style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", 
                       font=("Segoe UI", 10, "bold"),
                       padding=8,
                       background="#1976D2",
                       foreground="white",
                       borderwidth=0)
        style.map("TButton",
                 background=[("active", "#1565C0"), ("disabled", "#B0BEC5")],
                 foreground=[("disabled", "#78909C")])
        style.configure("TLabel", 
                       background="#F5F6F5",
                       foreground="#212121",
                       font=("Segoe UI", 10))
        style.configure("TFrame", 
                       background="#FFFFFF",
                       relief="flat",
                       borderwidth=1,
                       bordercolor="#E0E0E0")
        style.configure("TEntry",
                       fieldbackground="#FFFFFF",
                       font=("Segoe UI", 10),
                       padding=6,
                       borderwidth=1)
        style.configure("TProgressbar",
                       background="#1976D2",
                       troughcolor="#E0E0E0")

        # Agent and control variables
        self.agent = ResearchAgent()
        self.stop_event = threading.Event()

        # Header
        self.header_frame = ttk.Frame(root, style="TFrame")
        self.header_frame.pack(fill="x", pady=0)
        self.header_label = ttk.Label(self.header_frame,
                                    text="Research Agent",
                                    font=("Segoe UI", 16, "bold"),
                                    foreground="#1976D2")
        self.header_label.pack(pady=15)

        # Main container
        self.main_frame = ttk.Frame(root, padding=20, style="TFrame")
        self.main_frame.pack(fill="both", expand=True)

        # Input area
        self.input_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.input_frame.pack(fill="x", pady=(0, 15))
        self.topic_label = ttk.Label(self.input_frame,
                                   text="Research Topic",
                                   font=("Segoe UI", 11, "bold"))
        self.topic_label.pack(pady=(10, 5), padx=10, anchor="w")
        self.input_subframe = ttk.Frame(self.input_frame)
        self.input_subframe.pack(fill="x", padx=10, pady=(0, 10))
        self.topic_entry = ttk.Entry(self.input_subframe)
        self.topic_entry.pack(side="left", fill="x", expand=True)
        self.topic_entry.insert(0, "Tamilnadu")
        self.clear_button = ttk.Button(self.input_subframe,
                                     text="Clear",
                                     command=self.clear_input,
                                     width=10)
        self.clear_button.pack(side="left", padx=(10, 0))

        # Controls
        self.control_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.control_frame.pack(fill="x", pady=15)
        self.research_button = ttk.Button(self.control_frame,
                                        text="Start Research",
                                        command=self.run_research,
                                        width=15)
        self.research_button.pack(side="left", padx=10)
        self.stop_button = ttk.Button(self.control_frame,
                                    text="Stop",
                                    command=self.stop_research,
                                    width=15,
                                    state="disabled")
        self.stop_button.pack(side="left", padx=10)
        self.save_button = ttk.Button(self.control_frame,
                                    text="Save Output",
                                    command=self.save_output,
                                    width=15,
                                    state="disabled")
        self.save_button.pack(side="left", padx=10)

        # Status bar with progress
        self.status_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.status_frame.pack(fill="x", pady=15)
        self.status_label = ttk.Label(self.status_frame,
                                    text="Ready",
                                    foreground="#1976D2")
        self.status_label.pack(side="left", padx=10)
        self.progress = ttk.Progressbar(self.status_frame,
                                      length=200,
                                      mode="determinate",
                                      maximum=100)
        self.progress.pack(side="left", padx=10)
        self.spinner = ttk.Label(self.status_frame,
                                text="",
                                font=("Segoe UI", 12))
        self.spinner.pack(side="left")
        self.spinner_running = False

        # Output area
        self.output_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.output_frame.pack(fill="both", expand=True)
        self.output_label = ttk.Label(self.output_frame,
                                    text="Research Output",
                                    font=("Segoe UI", 11, "bold"))
        self.output_label.pack(pady=(10, 5), padx=10, anchor="w")
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame,
            height=20,
            wrap=tk.WORD,
            bg="#FFFFFF",
            fg="#212121",
            font=("Segoe UI", 10),
            borderwidth=1,
            relief="flat"
        )
        self.output_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.output_text.tag_configure("header", font=("Segoe UI", 11, "bold"))
        self.output_text.tag_configure("url", foreground="#1976D2")

    def clear_input(self):
        self.topic_entry.delete(0, tk.END)

    def run_research(self):
        self.output_text.delete(1.0, tk.END)
        self.status_label.config(text="Collecting data...", foreground="#F57C00")
        self.progress["value"] = 0
        self.start_spinner()
        self.research_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.save_button.config(state="disabled")
        
        self.stop_event.clear()
        topic = self.topic_entry.get().strip()
        if not topic:
            self.output_text.insert(tk.END, "Error: Please enter a topic.")
            self.reset_ui()
            return
        
        thread = threading.Thread(target=self.perform_research, args=(topic,))
        thread.start()

    def stop_research(self):
        self.stop_event.set()
        self.status_label.config(text="Stopping...", foreground="#D32F2F")

    def perform_research(self, topic):
        try:
            self.agent.collect_data(topic, self.stop_event)
            self.progress["value"] = 50  # Halfway after collection
            if self.stop_event.is_set():
                self.output_text.insert(tk.END, "Research stopped by user.\n")
                self.reset_ui()
                return
            
            self.status_label.config(text="Generating analysis...", foreground="#F57C00")
            response, sentiment = self.agent.analyze_and_generate(topic)
            self.progress["value"] = 100  # Complete after analysis
            
            self.output_text.insert(tk.END, f"Analysis for '{topic}' (Sentiment: {sentiment}):\n", "header")
            self.output_text.insert(tk.END, f"{response}\n\n")
            self.output_text.insert(tk.END, f"Sources ({len(self.agent.data)}):\n", "header")
            for item in self.agent.data:
                self.output_text.insert(tk.END, f"{item['url']} - Sentiment: {item['sentiment']}\n", "url")
            
            self.status_label.config(text="Completed", foreground="#1976D2")
            self.save_button.config(state="normal")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error: {str(e)}\n")
            self.status_label.config(text="Error occurred", foreground="#D32F2F")
        finally:
            self.reset_ui()

    def save_output(self):
        output = self.output_text.get(1.0, tk.END).strip()
        if output:
            filename = f"{self.topic_entry.get()}_research.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(output)
            self.status_label.config(text=f"Saved to {filename}", foreground="#1976D2")
            messagebox.showinfo("Success", f"Output saved to {filename}")
        else:
            self.status_label.config(text="No content to save", foreground="#D32F2F")

    def start_spinner(self):
        if not self.spinner_running:
            self.spinner_running = True
            def spin():
                chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                i = 0
                while self.spinner_running:
                    self.spinner.config(text=chars[i % len(chars)])
                    self.root.update()
                    threading.Event().wait(0.1)
                    i += 1
                self.spinner.config(text="")
            threading.Thread(target=spin, daemon=True).start()

    def reset_ui(self):
        self.spinner_running = False
        self.research_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.progress["value"] = 0

def main():
    root = tk.Tk()
    app = ResearchUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()