import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Define ANSI escape sequences for different colors
GREEN = "\033[92m"
RESET = "\033[0m"

def download_models():
    print(f"{GREEN}Downloading models...{RESET}")
    sys.stdout.flush()

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    print(f"{GREEN}Processor model downloaded.{RESET}")
    sys.stdout.flush()

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print(f"{GREEN}VisionEncoderDecoder model downloaded.{RESET}")
    sys.stdout.flush()

    model_dir = "./model/"
    processor.save_pretrained(model_dir)
    print(f"{GREEN}Processor model saved.{RESET}")
   

    model.save_pretrained(model_dir)
    print(f"{GREEN}VisionEncoderDecoder model saved.{RESET}")
   
    print(
       
     )
    print(f"{GREEN}Done .........{RESET}")

if __name__ == "__main__":
    download_models()
