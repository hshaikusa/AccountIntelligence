import os
import gradio as gr
from typing import List, Tuple
from dotenv import load_dotenv
import os.path as osp

from account_intelligence_ai.iteration1.factory import Iteration1Mode, create_chat_implementation as create_iteration1_chat


# Load environment variables
load_dotenv()

import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def create_demo(iteration: int = 1, mode_str: str = "part1"):
    """Create and return a Gradio demo with the specified iteration and mode.
    
    Args:
        iteration: Which iteration implementation to use (1, 2, or 3)
        mode_str: String representation of the mode ('part1', 'part2', or 'part3')
        
    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """

    if iteration == 1:
        # iteration 1 implementation
        # Convert string to enum
        mode_map = {
            "part1": Iteration1Mode.PART1_WEB_SEARCH,
            "part2": Iteration1Mode.PART2_DOCUMENT_RAG,
            "part3": Iteration1Mode.PART3_CORRECTIVE_RAG
        }
        
        if mode_str not in mode_map:
            raise ValueError(f"Unknown mode: {mode_str}. Choose from: {list(mode_map.keys())}")
        
        mode = mode_map[mode_str]
        
        # Initialize the chat implementation
        chat_interface = create_iteration1_chat(mode)
        
        # Create the Gradio interface with appropriate title based on mode
        titles = {
            "part1": "Account Intelligence Apowered by AI (Web Search)",
            "part2": "Account Intelligence powered by AI (RAG)",
            "part3": "Account Intelligence powered by AI (RAG + WebSearch)"
        }
        
        descriptions = {
            "part1": "Your Account intelligent AI assistant provides intelligent insights about the Company requested.",
            "part2": "Your Account intelligent AI assistant provides intelligent insights about the Company requested.",
            "part3": "Your Account intelligent AI assistant provides intelligent insights about the Company requested."
        }
    elif iteration == 2:
        # iteration 2 implementation
        pass
    elif iteration == 3:
        # iteration 3 implementation
        pass
    else:
        raise ValueError(f"Unknown iteration: {iteration}. Choose from: [1, 2, 3]")
    
    # Initialize the chat implementation
    chat_interface.initialize()
    
    # Create the respond function that uses our chat implementation
    def respond(message: str, history: List[Tuple[str, str]]) -> str:
        """Process the message and return a response.
        
        Args:
            message: The user's input message
            history: List of previous (user, assistant) message tuples
            
        Returns:
            str: The assistant's response
        """
        # Get response from our chat implementation
        return chat_interface.process_message(message, history)
    
    # Create the Gradio interface
    examples = [
        ["Provide an executive summary for NVIDIA?"],
        ["Provide strategic goals of NVIDIA?"],
    ]
    
    if iteration == 1 and mode_str == "part1":
        pass

    elif iteration == 1 and mode_str in ["part2", "part3"]:
        examples = [
            ["Provide an executive summary for NVIDIA?"],
            ["Provide strategic goals of NVIDIA?"]
        ]
    elif iteration == 2 and mode_str == "part1":
        pass
    elif iteration == 2 and mode_str == "part2":
        pass
    elif iteration == 2 and mode_str == "part3":
        pass
    elif iteration == 3 and mode_str == "part1":
        pass
    elif iteration == 3 and mode_str == "part2":
        pass
    elif iteration == 3 and mode_str == "part3":
        pass

    # Default logo path (Ensure this file exists)
    # Logo Path
    logo_file = "logo.png"  # Ensure the file exists in the working directory
    LOGO_DIR = "./"

    logo_path = osp.join(LOGO_DIR, logo_file),
    # Custom HTML & CSS for positioning the logo at the top-right
    base64_logo = encode_image("logo.png")

    logo_html = f"""
        <div style="position: fixed; top: 10px; left: 10px; z-index: 1000; overflow: hidden;">
            <img src="data:image/png;base64,{base64_logo}" style="width: 320px; max-width: 100%; height: auto; border-radius: 20px;">
        </div>
    """
    # # Custom HTML for credits at the bottom
    # credits_html = """
    #     <div style="position: fixed; bottom: 50px; width: 100%; text-align: center; color: #333;">
    #         <p>Credits: #problem-first-ai @Aish @Kirti -- "Account Intelligence AI ^2" - Developed by [Ali Hashim JK Karla Puja Srinithi Venkat]</p>
    #     </div>
    # """
    # Custom HTML for fancy and colorful credits below the prompt text box
    credits_html = """
        <div style="text-align: center; color: #333; margin-top: 30px;">
            <p style="font-size: 14px; color: #4CAF50; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                Credits: <span style="color: #1E90FF;">"#problem-first-ai @Aish @Kirti Account Intelligence AI ^2" --> Developed by Ali Hashim JK Karla Puja Srinithi Venkat</span>
            </p>
        </div>
    """
    # # Custom HTML for fancy and colorful credits below the prompt text box
    # credits_html = """
    #     <div style="text-align: center; color: #333; margin-top: 50px;">
    #         <p style="font-size: 18px; font-weight: bold; color: #4CAF50; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);">
    #             Credits: #problem-first-ai @Aish @Kirti "Account Intelligence AI ^2" --> Developed by <span style="color: #1E90FF;">Ali Hashim JK Karla Puja Srinithi Venkat</span>
    #         </p>
    #     </div>
    # """
    # Create Gradio UI
    with gr.Blocks() as demo:
        gr.HTML("""
            <style>
                /* Set background for the entire UI */
                body, .gradio-container { 
                    background-color: #ADD8E6 !important;  /* Light Blue */
                    color: black !important;  /* Ensure text is black */
                    font-family: 'Arial', sans-serif;  /* Clean font */
                }

                /* Set background for the Chat Interface */
                .gradio-container .chatbot {
                    background: rgba(255, 255, 255, 0.8) !important;  /* Semi-transparent white */
                    color: #333 !important; /* Ensure clear text visibility */
                    border-radius: 10px;  /* Rounded corners */
                    padding: 10px;  /* Add some padding */
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
                    overflow: visible !important;  /* Ensure content is visible */
                    height: auto !important;  /* Adjust height to fit content */                
                }

                /* Ensure input boxes & other UI elements match the theme */
                input, textarea, button { 
                    color: black !important;
                    border-radius: 5px;  /* Rounded corners */
                    border: 1px solid #ccc;  /* Light gray border */
                    padding: 5px;  /* Add some padding */ 
                }

                /* Style buttons within the chat interface */
                .gradio-container .chatbot button {
                    background-color: #ADD8E6 !important;  /* Light Blue background */
                    color: #000 !important;  /* Black text */
                    border: none;  /* Remove border */
                    cursor: pointer;  /* Pointer cursor on hover */
                    transition: background-color 0.3s;  /* Smooth transition */
                }

                button:hover {
                    background-color: #87CEEB;  /* Sky Blue on hover */
                }

                /* Textual effects */
                h1, h2, h3, h4, h5, h6 {
                    color: #2e8b57;  /* Sea Green */
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
                }

                p {
                    color: #333;  /* Dark Gray */
                    line-height: 1.6;  /* Improved readability */
                }

                a {
                    color: #1e90ff;  /* Dodger Blue */
                    text-decoration: none;  /* Remove underline */
                }

                a:hover {
                    text-decoration: underline;  /* Underline on hover */
                }                
            </style>
        """)
        # with gr.Blocks(css="body { background-color: #FF00FF; }") as demo:
        gr.HTML(logo_html)  # Inject custom styling
        # gr.HTML("""
        #         <style>
        #         body { background-color: #FF00FF; }  /* Light gray background */
        #         </style>
        #         """)

        # gr.Markdown("# Welcome to the App")  # Main Title

        gr.ChatInterface(
            fn=respond,
            title=titles[mode_str],
            type="messages",
            description=descriptions[mode_str],
            examples=examples,
            theme=gr.themes.Soft()
        )
        gr.HTML(credits_html)  # Inject credits at the bottom
    
    return demo

# For backward compatibility and direct imports
demo = create_demo(iteration=1, mode_str="part1")
