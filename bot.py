import logging
from enum import Enum
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as ort
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import cv2
from ultralytics import YOLO
import os
import telegram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOption(Enum):
    MULTICLASS = 'yolo_multiclass.onnx'
    ADIPOSYTES = 'yolo_culture1.onnx'
    FIBROBLASTS = 'yolo_culture2.onnx'
    MESENCHYMAL = 'yolo_culture3.onnx'
    @classmethod
    def to_buttons(cls):
        return [
            InlineKeyboardButton(option.name.replace('_', ' ').capitalize(), callback_data=f'option_{i+1}')
            for i, option in enumerate(cls)
        ]

def load_model(model_name):
    model_path = model_name
    model = YOLO(model_path, task='segment')

    return model

loaded_models = {option: load_model(option.value) for option in ModelOption}

class ImageProcessor:
    def process(self, model, image: Image.Image) -> Image.Image:
        raise NotImplementedError

def split_image(image, chunk_size=(640, 640), overlap=0.5):
    img_width, img_height = image.size
    chunk_width, chunk_height = chunk_size
    
    step_x = int(chunk_width * (1 - overlap))
    step_y = int(chunk_height * (1 - overlap))
    
    chunks = []
    
    for y in range(0, img_height - chunk_height + 1, step_y):
        for x in range(0, img_width - chunk_width + 1, step_x):
            box = (x, y, x + chunk_width, y + chunk_height)
            chunk = image.crop(box)
            chunks.append((x, y, chunk))
    
    return chunks

def combine_chunks(chunks, img_size, chunk_size=(640, 640), overlap=1.3):
    img_width, img_height = img_size
    chunk_width, chunk_height = chunk_size
    
    full_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    count_map = np.zeros((img_height, img_width), dtype=np.uint8)
    
    step_x = int(chunk_width * (1 - overlap))
    step_y = int(chunk_height * (1 - overlap))
    
    for (x, y, chunk) in chunks:
        chunk_array = np.array(chunk)
        
        if chunk_array.shape[0] != chunk_height or chunk_array.shape[1] != chunk_width:
            chunk_array = cv2.resize(chunk_array, (chunk_width, chunk_height))
        
        full_image[y:y+chunk_height, x:x+chunk_width] += chunk_array
        count_map[y:y+chunk_height, x:x+chunk_width] += 1
    
    full_image = full_image // count_map[:, :, None]
    
    return Image.fromarray(full_image)

class ModelImageProcessor(ImageProcessor):
    def process(self, model, image: Image.Image, use_multiclass: bool) -> tuple:
        image_cv2 = np.array(image)  
        cv2.imwrite("tmp.bmp", image_cv2)

        results = model("tmp.bmp", conf=0.001, iou=0.05, max_det=3000)

        class_counts = {}
        for result in results:
            if result.boxes is not None:
                class_ids = result.boxes.cls.int()
                for class_id in class_ids:
                    class_name = model.names[class_id.item()]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if use_multiclass and class_counts:
            most_common_class = max(class_counts, key=class_counts.get)
            most_common_count = class_counts[most_common_class]

            filtered_results = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    class_ids = boxes.cls.int()
                    keep_indices = [
                        i for i, class_id in enumerate(class_ids)
                        if model.names[class_id.item()] == most_common_class
                    ]

                    if keep_indices:
                        result.boxes = boxes[keep_indices]  
                        if result.masks is not None:
                            result.masks = result.masks[keep_indices]  
                        filtered_results.append(result)

            return filtered_results, most_common_class, most_common_count

        total_objects = sum(class_counts.values())
        return results, None, total_objects

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[button] for button in ModelOption.to_buttons()]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose a model for image processing:", reply_markup=reply_markup)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    callback_mapping = {
        'option_1': ModelOption.MULTICLASS,
        'option_2': ModelOption.ADIPOSYTES,
        'option_3': ModelOption.FIBROBLASTS,
        'option_4': ModelOption.MESENCHYMAL,
    }

    chosen_model = callback_mapping.get(query.data)

    if chosen_model:
        context.user_data['chosen_model'] = chosen_model
        await query.edit_message_text(f"Model chosen: {chosen_model.name.replace('_', ' ').capitalize()}. Now send an image for processing.")
    else:
        await query.edit_message_text("Invalid option selected. Please choose a valid model.")
async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if 'chosen_model' not in context.user_data:
        await update.message.reply_text("Please choose a model first using /start.")
        return

    chosen_model = context.user_data['chosen_model']
    model = loaded_models[chosen_model]

    file = await update.message.photo[-1].get_file()
    file_data = await file.download_as_bytearray()

    logger.info(f"Received file of size: {len(file_data)} bytes")

    try:
        image = Image.open(BytesIO(file_data))
        logger.info(f"Image loaded successfully: {image.size}, mode: {image.mode}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        await update.message.reply_text("Failed to process the image. Please try again.")
        return

    processor = ModelImageProcessor()
    use_multiclass = True if chosen_model == ModelOption.MULTICLASS else False

    results, most_common_class, object_count = processor.process(model, image, use_multiclass=use_multiclass)

    if use_multiclass and most_common_class:
        message = f"Cell culture detected: {most_common_class}\nThe number of cell detected: {object_count}"
    else:
        message = f"Total cells detected: {object_count}" if object_count > 0 else "No cells detected."

    await update.message.reply_text(message)

    if results:
        annotated_image = results[0].plot(labels=False, boxes=False, masks=True)
        image_pillow = Image.fromarray(annotated_image)

        bio = BytesIO()
        bio.name = 'processed_image.png'
        image_pillow.save(bio, 'PNG')
        bio.seek(0)
        await update.message.reply_photo(photo=bio)
TOKEN = os.getenv("TELEGRAM_TOKEN")
def main() -> None:
    application = Application.builder().token(TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, image_handler))

    # Start bot
    application.run_polling()

if __name__ == "__main__":
    main()
