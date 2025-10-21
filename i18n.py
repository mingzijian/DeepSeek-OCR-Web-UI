#!/usr/bin/env python3
"""
Internationalization module for OCR application
"""

# Language translations
TRANSLATIONS = {
    'en': {
        # Header and Title
        'app_title': '🔍 DeepSeek OCR Web UI',
        'app_description': 'Upload multiple images and use DeepSeek OCR model for text recognition. Supports custom prompts to control recognition behavior.',
        
        # Language switcher
        'language_label': '🌐 Language',
        'english': 'English',
        'chinese': '中文',
        
        # Upload section
        'upload_label': '📁 Upload Images',
        'prompt_label': '✏️ Custom Prompt',
        'prompt_placeholder': 'Enter OCR prompt, e.g.: Free OCR. or Convert the document to markdown.',
        
        # Preset buttons
        'preset_general': '📝 General OCR',
        'preset_markdown': '📄 Convert to Markdown',
        'preset_table': '📊 Table Recognition',
        
        # Action button
        'recognize_btn': '🚀 Start Recognition',
        
        # Results tabs
        'results_tab': 'Recognition Results',
        'summary_tab': 'Summary',
        
        # Results content
        'waiting_message': 'Waiting for image upload and recognition...',
        'waiting_summary': 'Waiting for recognition to complete...',
        'results_title': '# OCR Recognition Results\n\n',
        'no_result': '*No recognition result or empty result*',
        'summary_title': '📊 **Recognition Summary**\n\n',
        'total_images': 'Total images',
        'successful': 'Successfully recognized',
        'failed': 'Recognition failed',
        'success_rate': 'Success rate',
        
        # Status messages
        'loading_model': 'Loading model...',
        'processing_image': 'Processing image {current}/{total}...',
        'model_load_success': 'Model loaded successfully!',
        'model_load_failed': 'Model loading failed: {error}',
        'please_upload': 'Please upload at least one image',
        'empty_result': 'Model returned empty result',
        'empty_string': 'Model returned empty string',
        'recognition_failed': 'Recognition failed: {error}',
        
        # Instructions
        'instructions_title': '## 📋 Usage Instructions',
        'instructions_content': '''
1. **Upload Images**: Click the "Upload Images" area to select one or more image files
2. **Set Prompt**: 
   - Use preset buttons to quickly select common prompts
   - Or manually enter custom prompts
3. **Start Recognition**: Click the "Start Recognition" button and wait for model processing
4. **View Results**: Check recognition results and summary statistics on the right side

## 🔧 Prompt Instructions

- **Free OCR.**: General text recognition
- **<|grounding|>Convert the document to markdown.**: Convert document to Markdown format
- **<|grounding|>Extract all tables and convert to markdown format.**: Specialized for table recognition and conversion

## ⚙️ Model Configuration

- Model: DeepSeek-OCR
- Base size: 1024
- Image size: 640
- Crop mode: Enabled
        ''',
        
        # Error messages
        'file_not_exist': 'Image file does not exist: {path}',
        'cleanup_temp': 'Cleaning temporary file: {path}',
        'using_captured': 'Using captured output as result',
        'using_direct': 'Using directly returned result',
        'trying_captured': 'Trying to use captured output...',
        'raw_result_length': 'Raw result length: {length}',
        'cleaned_result': 'Cleaned result: {result}',
        'empty_after_clean': 'Result is empty after cleaning, using raw result',
    },
    
    'zh': {
        # Header and Title
        'app_title': '🔍 DeepSeek OCR 识别工具',
        'app_description': '上传多张图片，使用 DeepSeek OCR 模型进行文字识别。支持自定义提示词来控制识别行为。',
        
        # Language switcher
        'language_label': '🌐 语言',
        'english': 'English',
        'chinese': '中文',
        
        # Upload section
        'upload_label': '📁 上传图片',
        'prompt_label': '✏️ 自定义提示词',
        'prompt_placeholder': '输入OCR提示词，例如：Free OCR. 或 Convert the document to markdown.',
        
        # Preset buttons
        'preset_general': '📝 通用OCR',
        'preset_markdown': '📄 转换为Markdown',
        'preset_table': '📊 表格识别',
        
        # Action button
        'recognize_btn': '🚀 开始识别',
        
        # Results tabs
        'results_tab': '识别结果',
        'summary_tab': '识别摘要',
        
        # Results content
        'waiting_message': '等待上传图片并点击识别...',
        'waiting_summary': '等待识别完成...',
        'results_title': '# OCR 识别结果\n\n',
        'no_result': '*无识别结果或结果为空*',
        'summary_title': '📊 **识别摘要**\n\n',
        'total_images': '总图片数',
        'successful': '成功识别',
        'failed': '识别失败',
        'success_rate': '成功率',
        
        # Status messages
        'loading_model': '正在加载模型...',
        'processing_image': '正在处理第 {current}/{total} 张图片...',
        'model_load_success': '模型加载成功！',
        'model_load_failed': '模型加载失败: {error}',
        'please_upload': '请上传至少一张图片',
        'empty_result': '模型返回了空结果',
        'empty_string': '模型返回了空字符串',
        'recognition_failed': '识别失败: {error}',
        
        # Instructions
        'instructions_title': '## 📋 使用说明',
        'instructions_content': '''
1. **上传图片**: 点击"上传图片"区域，选择一张或多张图片文件
2. **设置提示词**: 
   - 使用预设按钮快速选择常用提示词
   - 或者手动输入自定义提示词
3. **开始识别**: 点击"开始识别"按钮，等待模型处理
4. **查看结果**: 在右侧查看识别结果和摘要统计

## 🔧 提示词说明

- **Free OCR.**: 通用文字识别
- **<|grounding|>Convert the document to markdown.**: 将文档转换为Markdown格式
- **<|grounding|>Extract all tables...**: 专门用于表格识别和转换

## ⚙️ 模型配置

- 模型: DeepSeek-OCR
- 基础尺寸: 1024
- 图片尺寸: 640
- 裁剪模式: 启用
        ''',
        
        # Error messages
        'file_not_exist': '图片文件不存在: {path}',
        'cleanup_temp': '清理临时文件: {path}',
        'using_captured': '使用捕获的输出作为结果',
        'using_direct': '使用直接返回的结果',
        'trying_captured': '尝试使用捕获输出...',
        'raw_result_length': '原始结果长度: {length}',
        'cleaned_result': '清理后结果: {result}',
        'empty_after_clean': '清理后结果为空，使用原始结果',
    }
}

class I18n:
    """Internationalization class"""
    
    def __init__(self, default_language='en'):
        self.current_language = default_language
    
    def set_language(self, language):
        """Set current language"""
        if language in TRANSLATIONS:
            self.current_language = language
        else:
            print(f"Warning: Language '{language}' not supported, using default")
    
    def get(self, key, **kwargs):
        """Get translated text"""
        if self.current_language not in TRANSLATIONS:
            self.current_language = 'en'
        
        translation = TRANSLATIONS[self.current_language].get(key, key)
        
        # Format string with kwargs if provided
        if kwargs:
            try:
                return translation.format(**kwargs)
            except (KeyError, ValueError):
                return translation
        
        return translation
    
    def get_all_for_language(self, language=None):
        """Get all translations for a specific language"""
        lang = language or self.current_language
        return TRANSLATIONS.get(lang, TRANSLATIONS['en'])

# Global i18n instance
i18n = I18n()