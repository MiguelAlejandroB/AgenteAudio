import asyncio
import base64
import io
import json
import traceback
import wave
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import os
import re
import random
import tempfile
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    get_twilio_turn_credentials,
    wait_for_item,
)
from gradio.utils import get_space
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
import pygame

# Configuración de FFmpeg (asegúrate de que esté en tu PATH)
# Si usas Windows, descomenta esta línea y ajusta la ruta:
# AudioSegment.converter = r"C:\ruta\a\ffmpeg\bin\ffmpeg.exe"

load_dotenv()

cur_dir = Path(__file__).parent
SAMPLE_RATE = 24000

# Inicializar pygame para audio
pygame.mixer.init()


class LocalEnglishTutorHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=SAMPLE_RATE,
            output_frame_size=480,
            input_sample_rate=SAMPLE_RATE,
        )
        self.output_queue = asyncio.Queue()
        self.audio_buffer = []
        # Inicializar recognizer local
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        # Añadir contexto de conversación
        self.conversation_history = []
        self.user_errors = []
        self.conversation_topics = [
            "What do you like to do in your free time?",
            "Tell me about your favorite hobby.",
            "What's your favorite food and why?",
            "Describe your perfect day.",
            "What are your goals for learning English?",
            "Tell me about your family.",
            "What's interesting about your city?",
            "What kind of music do you enjoy?",
            "Do you prefer movies or books? Why?",
            "What's something new you learned recently?"
        ]

    def copy(self):
        return LocalEnglishTutorHandler()

    def is_valid_speech(self, text):
        """Verifica si el texto transcrito parece ser habla humana válida"""
        if not text or len(text.strip()) < 3:
            return False
            
        # Filtrar comandos de dispositivos que se malinterpretan
        invalid_patterns = [
            'play', 'stop', 'pause', 'volume', 'alarm', 'timer', 'bluetooth', 
            'phone', 'call', 'message', 'weather', 'temperature', 'music',
            'search', 'google', 'assistant', 'siri', 'alexa', 'set', 'turn',
            'open', 'close', 'start', 'end', 'ok', 'hey', 'activate',
            'deactivate', 'on', 'off', 'up', 'down', 'left', 'right'
        ]
        
        text_lower = text.lower().strip()
        
        # Si es muy corto y contiene comandos, probablemente no es válido
        if len(text_lower.split()) <= 3:
            for pattern in invalid_patterns:
                if pattern in text_lower:
                    return False
        
        # Si contiene números solos, probablemente no es conversación
        words = text_lower.split()
        if len(words) <= 2 and any(word.isdigit() for word in words):
            return False
            
        # Si es solo una palabra y está en la lista de comandos
        if len(words) == 1 and words[0] in invalid_patterns:
            return False
            
        return True

    def transcribe_audio_local(self, audio_data):
        """Transcribe audio usando SpeechRecognition local con validación"""
        temp_file_path = None
        try:
            # Verificar que hay suficiente audio
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Calcular volumen promedio para verificar si hay sonido real
            volume = np.sqrt(np.mean(audio_array**2))
            if volume < 0.01:  # Muy bajo volumen, probablemente silencio
                return None
            
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            
            # Crear archivo temporal WAV con nombre único
            timestamp = str(int(time.time() * 1000))
            temp_file_path = os.path.join(tempfile.gettempdir(), f"audio_{timestamp}.wav")
            
            # Escribir archivo WAV
            with wave.open(temp_file_path, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(audio_bytes)
            
            # Pequeña pausa para asegurar que el archivo se cierre
            time.sleep(0.1)
            
            # Usar SpeechRecognition para transcribir
            with sr.AudioFile(temp_file_path) as source:
                # Ajustar para ruido ambiental más agresivamente
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                # Verificar que el audio tiene contenido
                if len(audio.frame_data) < 1000:  # Muy poco audio
                    return None
                
                try:
                    # Configurar reconocimiento más estricto
                    self.recognizer.energy_threshold = 4000
                    self.recognizer.dynamic_energy_threshold = True
                    
                    # Usar Google Speech Recognition (gratis) con configuración específica
                    text = self.recognizer.recognize_google(
                        audio, 
                        language='en-US',
                        show_all=False
                    )
                    
                    # Validar que el texto tiene sentido
                    if not self.is_valid_speech(text):
                        print(f"🚫 Audio rechazado por validación: '{text}'")
                        return None
                    
                    print(f"✅ Audio válido transcrito: '{text}'")
                    return text
                    
                except sr.UnknownValueError:
                    print("🔇 No se pudo entender el audio claramente")
                    return None
                except sr.RequestError as e:
                    print(f"❌ Error de servicio de reconocimiento: {e}")
                    return None
                        
        except Exception as e:
            print(f"❌ Error en transcripción local: {e}")
            return None
        finally:
            # Limpiar archivo temporal de manera segura
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    time.sleep(0.1)
                    os.remove(temp_file_path)
                except Exception as cleanup_error:
                    pass

    def analyze_english_errors(self, text):
        """Analiza errores comunes en inglés"""
        errors = []
        suggestions = []
        
        # Errores comunes de gramática básica
        common_errors = {
            r'\bi am good\b': 'I am well',
            r'\bmore better\b': 'better',
            r'\bmore easy\b': 'easier',
            r'\bmore fun\b': 'more enjoyable',
            r'\bcan able to\b': 'can' or 'am able to',
            r'\bwant that\b': 'want to',
            r'\bexplain me\b': 'explain to me',
            r'\bmake homework\b': 'do homework',
            r'\bmake a question\b': 'ask a question',
        }
        
        for pattern, correction in common_errors.items():
            if re.search(pattern, text, re.IGNORECASE):
                errors.append({
                    'type': 'grammar',
                    'found': re.search(pattern, text, re.IGNORECASE).group(),
                    'suggestion': correction
                })
        
        return errors

    def get_conversation_prompt(self, user_text, errors=None):
        """Crea un prompt conversacional específico para práctica de inglés"""
        base_prompt = """You are an friendly English conversation partner helping someone practice English. 
        You should:
        1. Respond naturally to their message
        2. Keep the conversation flowing
        3. Ask follow-up questions to encourage more speaking
        4. Be encouraging and supportive
        5. Occasionally provide gentle corrections when needed
        
        User said: "{user_text}"
        """
        
        if errors:
            corrections = "\n".join([f"Note: '{error['found']}' could be improved as '{error['suggestion']}'" for error in errors])
            base_prompt += f"\n\nGentle corrections to mention naturally: {corrections}"
        
        base_prompt += "\n\nPlease respond in a conversational, encouraging way (maximum 50 words):"
        
        return base_prompt.format(user_text=user_text)

    def generate_conversational_response(self, user_text, errors=None):
        """Genera respuestas conversacionales usando patrones locales"""
        user_text_lower = user_text.lower()
        
        # Respuestas para saludos
        if any(greeting in user_text_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            responses = [
                "Hello! It's great to hear from you. How are you doing today?",
                "Hi there! I'm excited to practice English with you. What would you like to talk about?",
                "Hey! Welcome to our conversation practice. How has your day been so far?",
                "Good to see you! I'm here to help you practice English. What's on your mind?"
            ]
            return random.choice(responses)
        
        # Respuestas para despedidas
        elif any(goodbye in user_text_lower for goodbye in ['goodbye', 'bye', 'see you', 'talk later']):
            responses = [
                "Goodbye! Great job practicing English today. Keep up the excellent work!",
                "See you later! You did wonderful today. Remember to practice regularly!",
                "Bye! It was lovely talking with you. Your English is improving!"
            ]
            return random.choice(responses)
        
        # Respuestas para agradecimientos
        elif any(thanks in user_text_lower for thanks in ['thank', 'thanks']):
            responses = [
                "You're very welcome! Keep practicing and you'll get even better.",
                "My pleasure! That's what I'm here for. What else would you like to discuss?",
                "No problem at all! Your effort to practice is admirable."
            ]
            return random.choice(responses)
        
        # Respuestas generales conversacionales
        else:
            # Añadir corrección si hay errores
            response_start = ""
            if errors:
                error_note = f"Great! Just a small note: instead of '{errors[0]['found']}', you could say '{errors[0]['suggestion']}'. But I understood you perfectly! "
                response_start = error_note
            
            # Respuestas que mantienen la conversación
            responses = [
                f"{response_start}That's really interesting! Can you tell me more about that?",
                f"{response_start}I'd love to hear more details. What do you think about it?",
                f"{response_start}That sounds fascinating! How did you get into that?",
                f"{response_start}Great point! What's your experience with that?",
                f"{response_start}I see! What made you think of that?",
                f"{response_start}That's cool! Have you always felt that way?",
                f"{response_start}Interesting perspective! What would you recommend to others?",
                f"{response_start}Nice! What's the most exciting part about that for you?"
            ]
            
            # Si no hay conversación previa, hacer una pregunta de tema
            if len(self.conversation_history) < 3:
                topic_questions = [
                    f"{response_start}That's great! {random.choice(self.conversation_topics)}",
                    f"{response_start}Excellent! Now, {random.choice(self.conversation_topics).lower()}",
                ]
                return random.choice(topic_questions)
            
            return random.choice(responses)

    async def process_audio(self, audio_data):
        try:
            print("🎤 Procesando audio localmente...")
            
            # Transcribir audio usando modelo local
            user_text = await asyncio.get_event_loop().run_in_executor(
                None, self.transcribe_audio_local, audio_data
            )

            if not user_text:
                print("🔇 No se detectó habla válida")
                await self.output_queue.put(AdditionalOutputs({
                    "type": "response.audio_transcript.done",
                    "transcript": "I didn't catch that clearly. Please speak a bit louder and clearer, or try saying 'hello' to start.",
                    "user_input": "",
                    "errors_detected": []
                }))
                return

            user_text = user_text.strip()
            print(f"👤 Usuario dijo: '{user_text}'")

            # Analizar errores de inglés
            errors = self.analyze_english_errors(user_text)
            if errors:
                print(f"📝 Errores detectados: {[f\"{e['found']} → {e['suggestion']}\" for e in errors]}")
            
            # Agregar a historial
            self.conversation_history.append({"role": "user", "content": user_text})
            if errors:
                self.user_errors.extend(errors)

            # Generar respuesta conversacional usando patrones locales
            response_text = self.generate_conversational_response(user_text, errors)

            # Asegurar que la respuesta no sea demasiado larga
            if len(response_text) > 200:
                response_text = response_text[:200] + "..."

            # Agregar respuesta al historial
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Limitar historial a las últimas 10 interacciones
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            print(f"🤖 Respuesta: '{response_text}'")
            
            await self.output_queue.put(AdditionalOutputs({
                "type": "response.audio_transcript.done",
                "transcript": response_text,
                "user_input": user_text,
                "errors_detected": errors
            }))

        except Exception as e:
            print(f"❌ Error procesando audio: {traceback.format_exc()}")
            await self.output_queue.put(AdditionalOutputs({
                "type": "response.audio_transcript.done",
                "transcript": "I'm having trouble with the audio. Please make sure you're speaking clearly in English and try again.",
                "user_input": "",
                "errors_detected": [],
                "error": str(e)
            }))

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        _, array = frame
        self.audio_buffer.extend(array.squeeze().tolist())
        
        # Procesar cada 2 segundos de audio
        if len(self.audio_buffer) >= SAMPLE_RATE * 2:
            audio_chunk = self.audio_buffer[:SAMPLE_RATE * 2]
            self.audio_buffer = self.audio_buffer[SAMPLE_RATE * 2:]
            asyncio.create_task(self.process_audio(audio_chunk))

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.audio_buffer:
            await self.process_audio(self.audio_buffer)
        self.audio_buffer = []


def update_chatbot(chatbot: list[dict], response):
    """Actualiza el chatbot con la respuesta del asistente"""
    if 'user_input' in response:
        # Agregar mensaje del usuario si está disponible
        chatbot.append({"role": "user", "content": response['user_input']})
    
    # Agregar respuesta del asistente
    assistant_response = response['transcript']
    
    # Agregar indicaciones de errores si existen
    if 'errors_detected' in response and response['errors_detected']:
        errors_text = " (Gentle correction noted)"
        assistant_response += errors_text
    
    chatbot.append({"role": "assistant", "content": assistant_response})
    return chatbot


chatbot = gr.Chatbot(label="🗣️ English Conversation Practice", height=400, type="messages")
latest_message = gr.Textbox(visible=False)

stream = Stream(
    LocalEnglishTutorHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=180 if get_space() else None,  # Más tiempo para conversaciones largas
)

app = FastAPI()
stream.mount(app)


@app.get("/")
async def home():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (cur_dir / "index.html").read_text(encoding='utf-8')
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


@app.get("/outputs")
async def get_outputs(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            output_data = output.args[0]
            data = {
                "role": "assistant", 
                "content": output_data.get('transcript', ''),
                "user_input": output_data.get('user_input', ''),
                "errors_detected": output_data.get('errors_detected', [])
            }
            yield f"event: output\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)