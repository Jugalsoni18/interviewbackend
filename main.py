from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import base64
import time
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS
import asyncio
import aiofiles

# Load environment variables
load_dotenv()

# Import your custom classes
try:
    from app import VoiceInterviewCoach, FaceMonitor
except ImportError:
    print("Warning: Could not import VoiceInterviewCoach or FaceMonitor")
    VoiceInterviewCoach = None
    FaceMonitor = None

app = FastAPI(
    title="AI Interview Coach API",
    description="AI-powered interview practice platform with real-time feedback",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Get API key from environment variable
API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize coach
coach = VoiceInterviewCoach() if VoiceInterviewCoach else None

# Global variable to store interview context
interview_context = {
    "questions": [],
    "answers": [],
    "feedback": [],
    "follow_ups": [],
    "job_role": "professional",
    "position_level": "mid-level"
}


# Pydantic Models for Request/Response Validation
class QuestionRequest(BaseModel):
    job_role: str = "data scientist"
    interview_type: str = "general"
    position_level: str = "mid-level"
    num_questions: int = 3


class QuestionResponse(BaseModel):
    questions: List[str]


class TextToSpeechRequest(BaseModel):
    text: str


class SummaryRequest(BaseModel):
    questions: List[str] = []
    answers: List[str] = []


class SummaryResponse(BaseModel):
    summary: str
    condensed_summary: str


# Root endpoint - serve HTML
@app.get("/")
async def index():
    return FileResponse("templates/index.html")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Interview Coach"}


# Generate interview questions
@app.post("/api/generate-questions", response_model=QuestionResponse)
async def generate_questions(request: QuestionRequest):
    global interview_context
    
    try:
        # Store context
        interview_context["job_role"] = request.job_role
        interview_context["interview_type"] = request.interview_type
        interview_context["position_level"] = request.position_level

        # Validate interview type
        valid_types = ["general", "behavioral", "technical", "case-study", "situational"]
        interview_type = request.interview_type if request.interview_type in valid_types else "general"

        # Generate prompt based on interview type
        prompts = {
            "behavioral": f"""Generate {request.num_questions} challenging but realistic behavioral interview questions for a {request.position_level} {request.job_role} role.
                Focus EXCLUSIVELY on past experiences. Each question MUST start with "Tell me about a time when..." or "Describe a situation where..."
                Phrase them as clear, conversational questions without numbers or commentary.""",
            
            "technical": f"""Generate {request.num_questions} challenging but realistic technical interview questions for a {request.position_level} {request.job_role} role.
                Focus on technical skills, knowledge, and problem-solving abilities.
                Phrase them as clear, conversational questions without numbers or commentary.""",
            
            "case-study": f"""Generate {request.num_questions} challenging case study questions for a {request.position_level} {request.job_role} role.
                Present hypothetical business scenarios to analyze and solve.
                Phrase them as clear, conversational questions without numbers or commentary.""",
            
            "situational": f"""Generate {request.num_questions} challenging situational questions for a {request.position_level} {request.job_role} role.
                Focus on hypothetical scenarios with "What would you do if..." or "How would you handle..."
                Phrase them as clear, conversational questions without numbers or commentary.""",
            
            "general": f"""Generate {request.num_questions} challenging general interview questions for a {request.position_level} {request.job_role} role.
                Include questions about experience, skills, and work style.
                Phrase them as clear, conversational questions without numbers or commentary."""
        }

        prompt = prompts.get(interview_type, prompts["general"])

        if not coach:
            raise HTTPException(status_code=500, detail="Interview coach not initialized")

        # Generate questions asynchronously
        response = await asyncio.to_thread(coach.model.generate_content, prompt)
        questions = [q.strip() for q in response.text.split("\n") if q.strip()]

        # Clean up questions
        clean_questions = []
        for q in questions:
            if q and any(q.startswith(prefix) for prefix in ["1.", "2.", "3.", "4.", "5.", "Question"]):
                q = q.split(". ", 1)[-1] if ". " in q else q.split(":", 1)[-1]
            clean_questions.append(q.strip())

        # Ensure enough questions
        while len(clean_questions) < request.num_questions:
            clean_questions.append(f"Can you tell me about your experience as a {request.job_role}?")

        questions = clean_questions[:request.num_questions]

        # Reset interview context
        interview_context.update({
            "questions": questions,
            "answers": [""] * len(questions),
            "feedback": [""] * len(questions),
            "follow_ups": [[] for _ in range(len(questions))]
        })

        return QuestionResponse(questions=questions)

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Process answer (audio transcription and feedback)
@app.post("/api/process-answer")
async def process_answer(
    audio: UploadFile = File(...),
    question: str = Form(...),
    question_index: int = Form(0),
    is_follow_up: bool = Form(False),
    follow_up_index: int = Form(0)
):
    global interview_context
    
    try:
        if not coach:
            raise HTTPException(status_code=500, detail="Interview coach not initialized")

        # Save uploaded audio file
        temp_dir = coach.temp_dir
        timestamp = int(time.time())
        temp_file_path = os.path.join(temp_dir, f"web_recording_{timestamp}.wav")
        
        # Use aiofiles for async file operations
        async with aiofiles.open(temp_file_path, 'wb') as f:
            content = await audio.read()
            await f.write(content)

        # Transcribe audio (run in thread pool to avoid blocking)
        transcription = await asyncio.to_thread(
            coach.transcribe_audio_with_gemini, 
            temp_file_path
        )

        if not transcription:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        # Get context
        job_role = interview_context.get("job_role", "professional")
        interview_type = interview_context.get("interview_type", "general")
        position_level = interview_context.get("position_level", "mid-level")

        if is_follow_up:
            # Generate follow-up feedback
            feedback_prompt = f"""Give brief (2-3 sentence) feedback on this follow-up answer for a {interview_type} interview:
                Follow-up Question: {question}
                Answer: {transcription}
                Focus on their response quality. Keep it constructive."""

            feedback_response = await asyncio.to_thread(
                coach.model.generate_content, 
                feedback_prompt
            )
            brief_feedback = feedback_response.text.strip()

            # Store follow-up data
            if not isinstance(interview_context.get("follow_up_answers"), list):
                interview_context["follow_up_answers"] = [[] for _ in range(len(interview_context["questions"]))]
            if not isinstance(interview_context.get("follow_up_feedbacks"), list):
                interview_context["follow_up_feedbacks"] = [[] for _ in range(len(interview_context["questions"]))]

            while len(interview_context["follow_up_answers"]) <= question_index:
                interview_context["follow_up_answers"].append([])
            while len(interview_context["follow_up_feedbacks"]) <= question_index:
                interview_context["follow_up_feedbacks"].append([])
            while len(interview_context["follow_up_answers"][question_index]) <= follow_up_index:
                interview_context["follow_up_answers"][question_index].append("")
            while len(interview_context["follow_up_feedbacks"][question_index]) <= follow_up_index:
                interview_context["follow_up_feedbacks"][question_index].append("")

            interview_context["follow_up_answers"][question_index][follow_up_index] = transcription
            interview_context["follow_up_feedbacks"][question_index][follow_up_index] = brief_feedback

            return {
                "transcription": transcription,
                "feedback": brief_feedback
            }
        
        else:
            # Generate detailed analysis for main question
            analysis_prompt = f"""Analyze this answer for a {interview_type} interview for a {position_level} {request.job_role} position:
                Question: {question}
                Answer: {transcription}
                
                Provide detailed feedback on:
                1. Content Quality (0-10)
                2. Relevance to Question (0-10)
                3. Completeness (0-10)
                4. Sentiment Analysis
                5. Strengths
                6. Areas for Improvement
                7. Specific Feedback
                8. Overall Impression
                9. Additional Insights"""

            # Generate follow-up question
            followup_prompt = f"""Based on this {interview_type} interview answer, generate 1 follow-up question:
                Question: {question}
                Answer: {transcription}
                Make it conversational and appropriate for {position_level} level."""

            # Run both prompts concurrently using asyncio.gather
            analysis_response, followup_response = await asyncio.gather(
                asyncio.to_thread(coach.model.generate_content, analysis_prompt),
                asyncio.to_thread(coach.model.generate_content, followup_prompt)
            )

            analysis = analysis_response.text.strip()
            follow_up_questions = [followup_response.text.strip()]

            # Extract feedback summary
            if "Specific Feedback:" in analysis:
                specific_feedback = analysis.split("Specific Feedback:")[1].strip()
                feedback_summary = f"Here's my feedback on your answer. {specific_feedback}"
            else:
                feedback_summary = "I've analyzed your answer and provided detailed feedback."

            # Update context
            while len(interview_context["answers"]) <= question_index:
                interview_context["answers"].append("")
            interview_context["answers"][question_index] = transcription

            while len(interview_context["feedback"]) <= question_index:
                interview_context["feedback"].append("")
            interview_context["feedback"][question_index] = analysis

            while len(interview_context["follow_ups"]) <= question_index:
                interview_context["follow_ups"].append([])
            interview_context["follow_ups"][question_index] = follow_up_questions

            # Save feedback to file asynchronously
            feedback_file = os.path.join(coach.temp_dir, f"feedback_q{question_index+1}_{timestamp}.txt")
            async with aiofiles.open(feedback_file, 'w') as f:
                await f.write(f"Question: {question}\n\n")
                await f.write(f"Your Answer: {transcription}\n\n")
                await f.write(f"Feedback:\n{analysis}")

            return {
                "transcription": transcription,
                "feedback": analysis,
                "feedback_summary": feedback_summary,
                "follow_up_questions": follow_up_questions
            }

    except Exception as e:
        print(f"Error processing answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Text-to-speech endpoint
@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="No text provided")

        if not coach:
            raise HTTPException(status_code=500, detail="Interview coach not initialized")

        timestamp = int(time.time())
        temp_file = os.path.join(coach.temp_dir, f"speech_{timestamp}.mp3")

        # Generate TTS asynchronously
        await asyncio.to_thread(
            lambda: gTTS(text=request.text, lang='en', slow=False).save(temp_file)
        )

        if not os.path.exists(temp_file):
            raise HTTPException(status_code=500, detail="Failed to create audio file")

        # Read and encode audio file
        async with aiofiles.open(temp_file, "rb") as audio_file:
            audio_content = await audio_file.read()
            encoded_audio = base64.b64encode(audio_content).decode('utf-8')

        # Clean up
        try:
            os.remove(temp_file)
        except:
            pass

        return {
            "success": True,
            "audio_data": encoded_audio,
            "mime_type": "audio/mp3"
        }

    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Generate interview summary
@app.post("/api/generate-summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    global interview_context
    
    try:
        questions = interview_context.get("questions", request.questions)
        answers = interview_context.get("answers", request.answers)
        feedbacks = interview_context.get("feedback", [])
        job_role = interview_context.get("job_role", "professional")
        interview_type = interview_context.get("interview_type", "general")
        position_level = interview_context.get("position_level", "mid-level")

        if not coach:
            raise HTTPException(status_code=500, detail="Interview coach not initialized")

        summary_prompt = f"""As an interview coach for a {position_level} {job_role} position, provide a comprehensive summary of this {interview_type} interview session.
            
            Questions asked:
            {chr(10).join(f"- {q}" for q in questions)}
            
            Summarize:
            1. Overall performance
            2. Key recommendations for {interview_type} interviews
            3. Preparation steps for future interviews
            
            Keep it concise but actionable."""

        condensed_prompt = f"""Create a brief (3-4 sentences) encouraging conclusion focusing on key strengths and one area to improve."""

        # Generate both summaries concurrently
        summary_response, condensed_response = await asyncio.gather(
            asyncio.to_thread(coach.model.generate_content, summary_prompt),
            asyncio.to_thread(coach.model.generate_content, condensed_prompt)
        )

        summary = summary_response.text.strip()
        condensed_summary = condensed_response.text.strip()

        # Save summary file asynchronously
        timestamp = int(time.time())
        summary_file = os.path.join(coach.temp_dir, f"interview_summary_{timestamp}.txt")
        async with aiofiles.open(summary_file, 'w') as f:
            await f.write(f"Interview Summary for {position_level.capitalize()} {job_role} - {interview_type.capitalize()}\n\n")
            await f.write(summary)
            await f.write("\n\nDetailed Records:\n\n")
            
            for q_idx, question in enumerate(questions):
                await f.write(f"Question {q_idx + 1}: {question}\n\n")
                if q_idx < len(answers):
                    await f.write(f"Answer: {answers[q_idx]}\n\n")
                if q_idx < len(feedbacks):
                    await f.write(f"Feedback: {feedbacks[q_idx]}\n\n")
                await f.write("-" * 50 + "\n\n")

        return SummaryResponse(
            summary=summary,
            condensed_summary=condensed_summary
        )

    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Face monitoring endpoints (placeholder - implement client-side)
@app.post("/api/start-face-monitoring")
async def start_face_monitoring():
    return {"success": True, "message": "Face monitoring should be implemented client-side"}


@app.post("/api/stop-face-monitoring")
async def stop_face_monitoring():
    return {"success": True}


# Cleanup endpoints
@app.post("/api/cleanup-question")
async def cleanup_question():
    return {"success": True}


@app.post("/api/cleanup")
async def cleanup():
    try:
        if coach:
            await asyncio.to_thread(coach.cleanup_audio_files)
        return {"success": True}
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# For local development
if __name__ == '__main__':
    import uvicorn
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting AI Interview Coach FastAPI Application...")
    if coach:
        print(f"Using temporary directory: {coach.temp_dir}")
    
    uvicorn.run(app, host="0.0.0.0", port=4000, reload=True)