import re
from datetime import datetime, timedelta
import cv2
import easyocr

class CCTVTimestampExtractor:
    def __init__(self, fps=30):
        # We initialize easyocr here. It will use GPU if available.
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.fps = fps
        self.last_valid_time = None
        self.last_frame_number = 0
        # Matches formats like HH:MM:SS, HH.MM.SS, HH;MM;SS
        self.time_pattern = re.compile(r'(\d{1,2})[:\.;](\d{2})[:\.;](\d{2})')

    def _extract_from_image(self, frame):
        h, w = frame.shape[:2]
        
        # Define common corners to search for timestamp (top-left, top-right, bottom-left, bottom-right)
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.25)
        
        regions = [
            frame[0:margin_h, 0:margin_w], # Top-Left
            frame[0:margin_h, w - margin_w:w], # Top-Right
            frame[h - margin_h:h, 0:margin_w], # Bottom-Left
            frame[h - margin_h:h, w - margin_w:w], # Bottom-Right
        ]
        
        for region in regions:
            # Upscale and grayscale to improve OCR
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            # Thresholding sometimes helps with digital clock fonts
            # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            results = self.reader.readtext(gray, detail=0)
            
            text = " ".join(results)
            match = self.time_pattern.search(text)
            if match:
                try:
                    time_str = f"{match.group(1)}:{match.group(2)}:{match.group(3)}"
                    parsed_time = datetime.strptime(time_str, "%H:%M:%S")
                    return parsed_time
                except ValueError:
                    continue
                    
        return None

    def get_timestamp(self, frame, frame_number):
        # Run OCR every 1 second (fps frames) or if we don't have a valid time yet
        if self.last_valid_time is None or frame_number % int(self.fps) == 0:
            extracted = self._extract_from_image(frame)
            if extracted:
                self.last_valid_time = extracted
                self.last_frame_number = frame_number
                return extracted.strftime("%H:%M:%S")
        
        # If OCR fails or we're in-between OCR checks, interpolate
        if self.last_valid_time is not None:
            frames_elapsed = frame_number - self.last_frame_number
            seconds_elapsed = frames_elapsed / self.fps
            current_time = self.last_valid_time + timedelta(seconds=seconds_elapsed)
            return current_time.strftime("%H:%M:%S")
            
        # Absolute fallback if we haven't found any timestamp yet
        seconds = int(frame_number / self.fps)
        return f"00:00:{seconds:02d}" if seconds < 60 else f"00:{seconds//60:02d}:{seconds%60:02d}"
