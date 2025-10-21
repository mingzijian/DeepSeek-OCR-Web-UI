import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import tempfile
from typing import List, Tuple
import base64
from PIL import Image
import traceback
from i18n import i18n
import sys
from io import StringIO
import re

class OCRApp:
    """OCR Application with DeepSeek-OCR model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Model path - Users need to modify this path according to their setup
        # Download from: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-OCR
        # or: https://huggingface.co/deepseek-ai/DeepSeek-OCR
        #self.model_path = '/you_path/deepseek-ai/DeepSeek-OCR'
        self.model_path = 'deepseek-ai/DeepSeek-OCR'
        
    def set_model_path(self, path):
        """Set custom model path"""
        self.model_path = path
        self.model_loaded = False  # Reset model loading status
        
    def load_model(self):
        """Load OCR model"""
        if self.model_loaded:
            return
            
        try:
            # Set GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            
            print(f"Loading model from: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                _attn_implementation='flash_attention_2', 
                trust_remote_code=True, 
                use_safetensors=True
            )
            self.model = self.model.eval().cuda().to(torch.bfloat16)
            self.model_loaded = True
            print(i18n.get('model_load_success'))
            
        except Exception as e:
            error_msg = i18n.get('model_load_failed', error=str(e))
            print(error_msg)
            traceback.print_exc()
            raise e
    
    def clean_ocr_result(self, raw_result: str) -> str:
        """Clean and format raw OCR result"""
        if not raw_result:
            return ""
        
        # Remove special markers and debug information
        lines = raw_result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip debug information lines
            if line.startswith('BASE:') or line.startswith('PATCHES:') or line.startswith('====='):
                continue
            # Skip warning messages
            if 'attention layers' in line or 'position_ids' in line or 'position_embeddings' in line:
                continue
            # Skip empty lines
            if not line:
                continue
                
            # Process lines with special markers
            if '<|ref|>' in line and '<|/ref|>' in line:
                # Extract reference content
                ref_pattern = r'<\|ref\|>(.*?)<\|/ref\|>'
                refs = re.findall(ref_pattern, line)
                if refs:
                    cleaned_lines.extend(refs)
                    
                # Remove all markers and get remaining content
                cleaned_line = re.sub(r'<\|[^|]*\|>', '', line).strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            else:
                # Remove other markers
                cleaned_line = re.sub(r'<\|[^|]*\|>', '', line)
                cleaned_line = re.sub(r'\[\[[^\]]*\]\]', '', cleaned_line).strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def process_images(self, images: List, prompt: str, progress=gr.Progress()) -> Tuple[str, str]:
        """Process multiple images for OCR recognition"""
        print(f"=== Starting image processing ===")
        print(f"Number of images: {len(images) if images else 0}")
        print(f"Prompt: {prompt}")
        
        if not images:
            return i18n.get('please_upload'), ""
            
        if not self.model_loaded:
            try:
                progress(0.1, desc=i18n.get('loading_model'))
                self.load_model()
            except Exception as e:
                error_msg = i18n.get('model_load_failed', error=str(e))
                print(error_msg)
                return error_msg, ""
        
        results = []
        total_images = len(images)
        
        for idx, image in enumerate(images):
            try:
                print(f"\n--- Processing image {idx + 1}/{total_images} ---")
                progress((idx + 1) / total_images, 
                        desc=i18n.get('processing_image', current=idx + 1, total=total_images))
                
                # Check image type
                print(f"Image type: {type(image)}")
                print(f"Image value: {image}")
                
                # Handle Gradio file objects
                if hasattr(image, 'name'):
                    # Gradio File object, use its path directly
                    temp_image_path = image.name
                    cleanup_temp = False
                    print(f"Using Gradio file path: {temp_image_path}")
                else:
                    # Create temporary file for other cases
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        # Save PIL Image object to file
                        if hasattr(image, 'save'):
                            image.save(temp_file.name)
                            print(f"Saved PIL image to: {temp_file.name}")
                        else:
                            # Copy file if it's a file path
                            import shutil
                            shutil.copy(image, temp_file.name)
                            print(f"Copied file to: {temp_file.name}")
                        
                        temp_image_path = temp_file.name
                        cleanup_temp = True
                
                # Check if file exists
                if not os.path.exists(temp_image_path):
                    raise Exception(i18n.get('file_not_exist', path=temp_image_path))
                
                print(f"Image file size: {os.path.getsize(temp_image_path)} bytes")
                
                # Execute OCR recognition
                formatted_prompt = f"<image>\n{prompt}"
                print(f"Formatted prompt: {formatted_prompt}")
                
                print("Starting OCR inference...")
                
                # Capture standard output to get streaming results
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                try:
                    result = self.model.infer(
                        self.tokenizer,
                        prompt=formatted_prompt,
                        image_file=temp_image_path,
                        output_path='./',
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        test_compress=True
                    )
                finally:
                    # Restore standard output
                    sys.stdout = old_stdout
                
                # Get captured output
                captured_text = captured_output.getvalue()
                
                # Use captured output if direct result is empty
                raw_result = ""
                if (result is None or str(result).strip() == "") and captured_text.strip():
                    raw_result = captured_text.strip()
                    print(f"Using captured output: {raw_result[:200]}...")
                elif result:
                    raw_result = str(result)
                    print(f"Using direct result: {raw_result[:200]}...")
                else:
                    print(f"Captured output: {captured_text[:200]}...")
                    if captured_text.strip():
                        raw_result = captured_text.strip()
                
                # Clean and format result
                cleaned_result = self.clean_ocr_result(raw_result)
                
                # Validate cleaned result
                if not cleaned_result or cleaned_result.strip() == "":
                    if raw_result:
                        # Use raw result if cleaned result is empty but raw result is not
                        cleaned_result = raw_result
                        print(i18n.get('empty_after_clean'))
                    else:
                        cleaned_result = i18n.get('empty_result')
                
                # Print debug information
                print(i18n.get('raw_result_length', length=len(raw_result)))
                print(i18n.get('cleaned_result', 
                              result=cleaned_result[:200] + "..." if len(cleaned_result) > 200 else cleaned_result))
                
                results.append({
                    'image_index': idx + 1,
                    'result': str(cleaned_result),
                    'status': 'success'
                })
                
                # Clean up temporary files (only if we created them)
                if cleanup_temp:
                    try:
                        os.unlink(temp_image_path)
                        print(i18n.get('cleanup_temp', path=temp_image_path))
                    except:
                        pass
                
            except Exception as e:
                error_msg = i18n.get('recognition_failed', error=str(e))
                print(f"Error: {error_msg}")
                traceback.print_exc()
                
                results.append({
                    'image_index': idx + 1,
                    'result': error_msg,
                    'status': 'error'
                })
        
        print(f"\n=== Processing complete, total results: {len(results)} ===")
        
        # Format results for display
        formatted_results = self.format_results(results)
        summary = self.generate_summary(results)
        
        print(f"Formatted results length: {len(formatted_results)}")
        print(f"Summary length: {len(summary)}")
        
        return formatted_results, summary
    
    def format_results(self, results: List[dict]) -> str:
        """Format recognition results for display"""
        formatted = i18n.get('results_title')
        
        for result in results:
            status_icon = "✅" if result['status'] == 'success' else "❌"
            formatted += f"## {i18n.get('results_tab')} {result['image_index']} {status_icon}\n\n"
            
            # Handle result content
            result_content = result['result']
            if not result_content or str(result_content).strip() == "":
                result_content = i18n.get('no_result')
            
            # Add collapsible display for long results
            if len(str(result_content)) > 1000:
                formatted += f"<details>\n<summary>View full result (Length: {len(str(result_content))} characters)</summary>\n\n"
                formatted += f"```\n{result_content}\n```\n\n"
                formatted += "</details>\n\n"
            else:
                formatted += f"```\n{result_content}\n```\n\n"
            
            formatted += "---\n\n"
        
        return formatted
    
    def generate_summary(self, results: List[dict]) -> str:
        """Generate recognition summary"""
        total = len(results)
        success = sum(1 for r in results if r['status'] == 'success')
        failed = total - success
        
        summary = i18n.get('summary_title')
        summary += f"- {i18n.get('total_images')}: {total}\n"
        summary += f"- {i18n.get('successful')}: {success}\n"
        summary += f"- {i18n.get('failed')}: {failed}\n"
        summary += f"- {i18n.get('success_rate')}: {(success/total*100):.1f}%\n\n"
        
        return summary

# Create OCR application instance
ocr_app = OCRApp()

# Language change handler
def change_language(language):
    """Handle language change"""
    i18n.set_language(language)
    print(f"Language changed to: {language}")
    return create_interface_components()

# Preset prompt handlers
def set_preset_general():
    return "Free OCR."

def set_preset_markdown():
    return "<|grounding|>Convert the document to markdown."

def set_preset_table():
    return "<|grounding|>Extract all tables and convert to markdown format."

def create_interface_components():
    """Create interface components with current language"""
    return (
        # Update all component labels and values
        gr.update(label=i18n.get('upload_label')),
        gr.update(label=i18n.get('prompt_label'), 
                 placeholder=i18n.get('prompt_placeholder')),
        gr.update(value=i18n.get('preset_general')),
        gr.update(value=i18n.get('preset_markdown')),
        gr.update(value=i18n.get('preset_table')),
        gr.update(value=i18n.get('recognize_btn')),
        gr.update(label=i18n.get('results_tab')),
        gr.update(label=i18n.get('summary_tab')),
        gr.update(value=i18n.get('waiting_message')),
        gr.update(value=i18n.get('waiting_summary')),
        gr.update(value=i18n.get('instructions_content'))
    )

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    with gr.Blocks(title=i18n.get('app_title'), theme=gr.themes.Soft()) as demo:
        # Language switcher at the top
        with gr.Row():
            gr.Markdown(f"# {i18n.get('app_title')}")
            with gr.Column(scale=1, min_width=150):
                language_selector = gr.Dropdown(
                    choices=[('English', 'en'), ('中文', 'zh')],
                    value='en',
                    label=i18n.get('language_label'),
                    interactive=True
                )
        
        gr.Markdown(i18n.get('app_description'))
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload area
                images_input = gr.File(
                    label=i18n.get('upload_label'),
                    file_count="multiple",
                    file_types=["image"],
                    height=200
                )
                
                # Prompt input
                prompt_input = gr.Textbox(
                    label=i18n.get('prompt_label'),
                    value="<|grounding|>Convert the document to markdown.",
                    placeholder=i18n.get('prompt_placeholder'),
                    lines=3
                )
                
                # Preset prompt buttons
                with gr.Row():
                    preset_ocr = gr.Button(i18n.get('preset_general'), size="sm")
                    preset_markdown = gr.Button(i18n.get('preset_markdown'), size="sm")
                    preset_table = gr.Button(i18n.get('preset_table'), size="sm")
                
                # Recognition button
                recognize_btn = gr.Button(
                    i18n.get('recognize_btn'), 
                    variant="primary", 
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Results display area
                with gr.Tab(i18n.get('results_tab')):
                    results_output = gr.Markdown(
                        label=i18n.get('results_tab'),
                        value=i18n.get('waiting_message'),
                        height=500
                    )
                
                with gr.Tab(i18n.get('summary_tab')):
                    summary_output = gr.Markdown(
                        label=i18n.get('summary_tab'),
                        value=i18n.get('waiting_summary'),
                        height=500
                    )
        
        # Instructions section
        instructions_md = gr.Markdown(i18n.get('instructions_content'))
        
        # Language change event
        language_selector.change(
            fn=lambda lang: change_language(lang),
            inputs=[language_selector],
            outputs=[
                images_input, prompt_input, preset_ocr, preset_markdown, 
                preset_table, recognize_btn, results_output, summary_output,
                results_output, summary_output, instructions_md
            ]
        )
        
        # Preset prompt click events
        preset_ocr.click(fn=set_preset_general, outputs=prompt_input)
        preset_markdown.click(fn=set_preset_markdown, outputs=prompt_input)
        preset_table.click(fn=set_preset_table, outputs=prompt_input)
        
        # Recognition button click event
        recognize_btn.click(
            fn=ocr_app.process_images,
            inputs=[images_input, prompt_input],
            outputs=[results_output, summary_output],
            show_progress=True
        )
    
    return demo

# Launch application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
