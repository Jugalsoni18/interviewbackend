from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import base64
import json
import threading
import time
from dotenv import load_dotenv
import google.generativeai as genai
from gtts import gTTS

# Load environment variables
load_dotenv()

# Import your custom classes (make sure app.py exists with these classes)
try:
    from app import VoiceInterviewCoach, FaceMonitor
except ImportError:
    print("Warning: Could not import VoiceInterviewCoach or FaceMonitor")
    VoiceInterviewCoach = None
    FaceMonitor = None

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, replace with your frontend domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Get API key from environment variable
API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize coach only if the class is available
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

# Flag to track if face monitoring is active
face_monitoring_active = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    global interview_context
    data = request.json
    job_role = data.get('job_role', 'data scientist')
    interview_type = data.get('interview_type', 'general')
    position_level = data.get('position_level', 'mid-level')
    num_questions = data.get('num_questions', 3)

    try:
        # Store the job role, interview type, and position level in the interview context
        interview_context["job_role"] = job_role
        interview_context["interview_type"] = interview_type
        interview_context["position_level"] = position_level

        # Validate interview type
        valid_interview_types = ["general", "behavioral", "technical", "case-study", "situational"]
        if interview_type not in valid_interview_types:
            interview_type = "general"

        # Use the interview type to customize the prompt
        if interview_type == "behavioral":
            prompt = f"""Generate {num_questions} challenging but realistic behavioral interview questions for a {position_level} {job_role} role.
                Focus EXCLUSIVELY on past experiences and how the candidate handled specific situations.
                Each question MUST start with phrases like "Tell me about a time when..." or "Describe a situation where..." or "Give me an example of..."
                The questions should assess how the candidate has demonstrated relevant skills in past roles.
                Adjust the complexity and expectations based on the {position_level} position level.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
        elif interview_type == "technical":
            prompt = f"""Generate {num_questions} challenging but realistic technical interview questions for a {position_level} {job_role} role.
                Focus EXCLUSIVELY on technical skills, knowledge, and problem-solving abilities specific to the {job_role} position.
                Questions should assess technical competence, domain knowledge, and analytical thinking.
                Include questions about methodologies, tools, and technologies relevant to a {job_role}.
                Adjust the technical depth and complexity based on the {position_level} position level.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
        elif interview_type == "case-study":
            prompt = f"""Generate {num_questions} challenging but realistic case study interview questions for a {position_level} {job_role} role.
                Present hypothetical business scenarios that the candidate needs to analyze and solve.
                Each question should describe a specific business problem or situation relevant to a {job_role} position.
                Focus on assessing problem-solving approach, analytical thinking, and business acumen.
                Adjust the complexity of the scenarios based on the {position_level} position level.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
        elif interview_type == "situational":
            prompt = f"""Generate {num_questions} challenging but realistic situational interview questions for a {position_level} {job_role} role.
                Focus EXCLUSIVELY on hypothetical future scenarios with questions like "What would you do if..." or "How would you handle..."
                Questions should assess decision-making, judgment, and how candidates would respond to job-specific situations.
                Make the scenarios relevant to challenges commonly faced in a {position_level} {job_role} position.
                Adjust the complexity and responsibility level based on the {position_level} position level.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
        else:  # general
            prompt = f"""Generate {num_questions} challenging but realistic general interview questions for a {position_level} {job_role} role.
                Include a balanced mix of questions about experience, skills, and work style.
                Cover topics like professional background, strengths and weaknesses, and career goals.
                Focus on questions that help assess overall fit for a {position_level} {job_role} position.
                Adjust the expectations and complexity based on the {position_level} position level.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""

        # Use the coach's model to generate questions
        if not coach:
            return jsonify({"error": "Interview coach not initialized"}), 500
            
        response = coach.model.generate_content(prompt)
        questions = [q.strip() for q in response.text.split("\n") if q.strip()]

        # Clean up questions
        clean_questions = []
        for q in questions:
            # Remove numbering if present
            if q and any(q.startswith(prefix) for prefix in ["1.", "2.", "3.", "4.", "5.", "Question"]):
                q = q.split(". ", 1)[-1] if ". " in q else q.split(":", 1)[-1]
            clean_questions.append(q.strip())

        # Make sure we have enough questions
        while len(clean_questions) < num_questions:
            clean_questions.append(f"Can you tell me about your experience as a {job_role}?")

        questions = clean_questions[:num_questions]

        # Reset interview context
        interview_context.update({
            "questions": questions,
            "answers": [""] * len(questions),
            "feedback": [""] * len(questions),
            "follow_ups": [[] for _ in range(len(questions))]
        })

        return jsonify({"questions": questions})

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/process-answer', methods=['POST'])
def process_answer():
    global interview_context
    try:
        audio_file = request.files.get('audio')
        question = request.form.get('question')
        question_index = int(request.form.get('question_index', 0))
        is_follow_up = request.form.get('is_follow_up') == 'true'
        follow_up_index = int(request.form.get('follow_up_index', 0)) if is_follow_up else 0

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        if not coach:
            return jsonify({"error": "Interview coach not initialized"}), 500

        # Create a temporary file to save the audio
        temp_dir = coach.temp_dir
        timestamp = int(time.time())
        temp_file_path = os.path.join(temp_dir, f"web_recording_{timestamp}.wav")
        audio_file.save(temp_file_path)

        # Transcribe the audio
        transcription = coach.transcribe_audio_with_gemini(temp_file_path)

        if not transcription:
            return jsonify({"error": "Failed to transcribe audio"}), 500

        # Get the job role, interview type, and position level from the interview context
        job_role = interview_context.get("job_role", "professional")
        interview_type = interview_context.get("interview_type", "general")
        position_level = interview_context.get("position_level", "mid-level")

        if is_follow_up:
            # For follow-up questions, customize feedback based on interview type
            if interview_type == "behavioral":
                brief_feedback_prompt = f"""Give a very brief (2-3 sentence) feedback on this follow-up answer for a behavioral interview:

                Follow-up Question: {question}
                Answer: {transcription}

                Ignore "*" when speaking.
                Focus on how well they elaborated on their specific actions and results in the situation.
                Keep it constructive and specific, mentioning one strength and one suggestion for improving their behavioral example.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
            elif interview_type == "technical":
                brief_feedback_prompt = f"""Give a very brief (2-3 sentence) feedback on this follow-up answer for a technical interview:

                Follow-up Question: {question}
                Answer: {transcription}

                Ignore "*" when speaking.
                Focus on the technical accuracy, depth of knowledge, and clarity of explanation.
                Keep it constructive and specific, mentioning one strength and one suggestion for improving their technical response.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
            elif interview_type == "case-study":
                brief_feedback_prompt = f"""Give a very brief (2-3 sentence) feedback on this follow-up answer for a case study interview:

                Follow-up Question: {question}
                Answer: {transcription}

                Ignore "*" when speaking.
                Focus on their analytical approach, problem-solving skills, and consideration of key factors.
                Keep it constructive and specific, mentioning one strength and one suggestion for improving their case analysis.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
            elif interview_type == "situational":
                brief_feedback_prompt = f"""Give a very brief (2-3 sentence) feedback on this follow-up answer for a situational interview:

                Follow-up Question: {question}
                Answer: {transcription}

                Ignore "*" when speaking.
                Focus on their decision-making process, judgment, and how they would handle the hypothetical scenario.
                Keep it constructive and specific, mentioning one strength and one suggestion for improving their approach.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""
            else:  # general
                brief_feedback_prompt = f"""Give a very brief (2-3 sentence) feedback on this follow-up answer for a general interview:

                Follow-up Question: {question}
                Answer: {transcription}

                Ignore "*" when speaking.
                Keep it constructive and specific, mentioning one strength and one suggestion for improvement.
                Phrase them as clear, conversational questions without numbers or commentary and start directly from question 1 without telling me what you're gonna do."""

            brief_feedback_response = coach.model.generate_content(brief_feedback_prompt)
            brief_feedback = brief_feedback_response.text.strip()

            # Store the follow-up answer and feedback
            if not isinstance(interview_context.get("follow_up_answers"), list):
                interview_context["follow_up_answers"] = [[] for _ in range(len(interview_context["questions"]))]

            if not isinstance(interview_context.get("follow_up_feedbacks"), list):
                interview_context["follow_up_feedbacks"] = [[] for _ in range(len(interview_context["questions"]))]

            # Extend lists if needed
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

            return jsonify({
                "transcription": transcription,
                "feedback": brief_feedback
            })
        else:
            # For main questions, generate detailed analysis
            if interview_type == "behavioral":
                analysis_prompt = f"""Analyze this answer for a behavioral interview for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Provide detailed feedback on:
                1. **Content Quality (0-10):** Rate and explain how well they described a specific situation and their role in it.
                2. **Relevance to Question (0-10):** How well did they address the specific behavioral scenario asked about?
                3. **Completeness (0-10):** Did they cover all elements of the STAR method (Situation, Task, Action, Result)?
                4. **Sentiment Analysis:** Comment on confidence, enthusiasm, and tone when describing past experiences.
                5. **Strengths:** What did they do well in describing their past behavior and actions?
                6. **Areas for Improvement:** How could they better structure their behavioral examples?
                7. **Specific Feedback:** Give actionable advice on improving their behavioral storytelling.
                8. **Overall Impression:** Summarize your assessment of their behavioral response.
                9. **Additional Insights:** Any other observations about how they presented their past experiences?
                10. don't add "*" when generating summary"""
            elif interview_type == "technical":
                analysis_prompt = f"""Analyze this answer for a technical interview for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Provide detailed feedback on:
                1. **Content Quality (0-10):** Rate and explain the technical accuracy and depth of knowledge shown.
                2. **Relevance to Question (0-10):** How well did they address the specific technical concepts asked about?
                3. **Completeness (0-10):** Did they cover all technical aspects of the question?
                4. **Sentiment Analysis:** Comment on confidence and clarity when explaining technical concepts.
                5. **Strengths:** What technical knowledge or skills did they demonstrate well?
                6. **Areas for Improvement:** What technical concepts could they explain better?
                7. **Specific Feedback:** Give actionable advice on improving their technical explanations.
                8. **Overall Impression:** Summarize your assessment of their technical knowledge.
                9. **Additional Insights:** Any other observations about their technical aptitude?
                10. don't add "*" when generating summary"""
            elif interview_type == "case-study":
                analysis_prompt = f"""Analyze this answer for a case study interview for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Provide detailed feedback on:
                1. **Content Quality (0-10):** Rate and explain their problem-solving approach and business acumen.
                2. **Relevance to Question (0-10):** How well did they address the specific case scenario presented?
                3. **Completeness (0-10):** Did they consider all important factors in their analysis?
                4. **Sentiment Analysis:** Comment on confidence and structured thinking when analyzing the case.
                5. **Strengths:** What aspects of their case analysis were effective?
                6. **Areas for Improvement:** How could they improve their case analysis approach?
                7. **Specific Feedback:** Give actionable advice on improving their case study responses.
                8. **Overall Impression:** Summarize your assessment of their analytical abilities.
                9. **Additional Insights:** Any other observations about their problem-solving process?
                10. don't add "*" when generating summary"""
            elif interview_type == "situational":
                analysis_prompt = f"""Analyze this answer for a situational interview for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Provide detailed feedback on:
                1. **Content Quality (0-10):** Rate and explain their approach to the hypothetical situation.
                2. **Relevance to Question (0-10):** How well did they address the specific scenario presented?
                3. **Completeness (0-10):** Did they consider all important aspects of the situation?
                4. **Sentiment Analysis:** Comment on confidence and decision-making clarity.
                5. **Strengths:** What aspects of their situational response were effective?
                6. **Areas for Improvement:** How could they better handle similar situations?
                7. **Specific Feedback:** Give actionable advice on improving their situational responses.
                8. **Overall Impression:** Summarize your assessment of their judgment and adaptability.
                9. **Additional Insights:** Any other observations about their approach to hypothetical scenarios?
                10. don't add "*" when generating summary"""
            else:  # general
                analysis_prompt = f"""Analyze this answer for a general interview for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Provide detailed feedback on:
                1. **Content Quality (0-10):** Rate and explain the substance of the answer.
                2. **Relevance to Question (0-10):** How well did they address what was asked?
                3. **Completeness (0-10):** Did they cover all necessary aspects?
                4. **Sentiment Analysis:** Comment on confidence, enthusiasm, and tone.
                5. **Strengths:** What did they do well?
                6. **Areas for Improvement:** What could be better?
                7. **Specific Feedback:** Give actionable advice.
                8. **Overall Impression:** Summarize your assessment.
                9. **Additional Insights:** Any other observations?
                10. don't add "*" when generating summary"""

            analysis_response = coach.model.generate_content(analysis_prompt)
            analysis = analysis_response.text.strip()

            # Generate follow-up questions
            if interview_type == "behavioral":
                followup_prompt = f"""Based on this behavioral interview question and answer for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Generate 1 follow-up question that:
                1. Probes deeper into the specific situation the candidate described
                2. Asks about their actions, the results, or lessons learned
                3. Follows the STAR (Situation, Task, Action, Result) method
                4. Starts with phrases like "Can you tell me more about..." or "What specifically did you..."
                5. Is appropriate for a {position_level} position level

                Phrase as a clear, conversational question without numbering or commentary.
                don't add "*" when generating summary"""
            elif interview_type == "technical":
                followup_prompt = f"""Based on this technical interview question and answer for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Generate 1 follow-up question that:
                1. Probes deeper into their technical knowledge or approach
                2. Asks about specific methodologies, tools, or technologies mentioned
                3. Challenges them to explain their reasoning or consider alternative approaches
                4. Tests deeper technical understanding relevant to a {job_role}
                5. Is appropriate for a {position_level} position level

                Phrase as a clear, conversational question without numbering or commentary.
                don't add "*" when generating summary"""
            elif interview_type == "case-study":
                followup_prompt = f"""Based on this case study interview question and answer for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Generate 1 follow-up question that:
                1. Adds complexity or a new constraint to the scenario
                2. Asks how they would handle a specific challenge within their proposed solution
                3. Probes their analytical thinking or business judgment
                4. Explores trade-offs or alternative approaches to the case
                5. Is appropriate for a {position_level} position level

                Phrase as a clear, conversational question without numbering or commentary.
                don't add "*" when generating summary"""
            elif interview_type == "situational":
                followup_prompt = f"""Based on this situational interview question and answer for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Generate 1 follow-up question that:
                1. Introduces a complication or twist to the hypothetical scenario
                2. Asks "What if..." or "How would you handle..." to explore their adaptability
                3. Probes their decision-making process or priorities
                4. Tests how they would respond to challenges specific to a {job_role}
                5. Is appropriate for a {position_level} position level

                Phrase as a clear, conversational question without numbering or commentary.
                don't add "*" when generating summary"""
            else:  # general
                followup_prompt = f"""Based on this general interview question and answer for a {position_level} {job_role} position:

                Question: {question}
                Answer: {transcription}

                Generate 1 follow-up question that:
                1. Probes deeper into areas that need more detail
                2. Explores their experience, skills, or work style further
                3. Helps assess the candidate's suitability for a {job_role} role
                4. Is appropriate for a {position_level} position level

                Phrase as a clear, conversational question without numbering or commentary.
                don't add "*" when generating summary"""

            followup_response = coach.model.generate_content(followup_prompt)
            follow_up_questions = [followup_response.text.strip()]

            # Extract feedback summary for TTS
            if "Specific Feedback:" in analysis:
                specific_feedback = analysis.split("Specific Feedback:")[1].strip()
                feedback_summary = f"Here's my feedback on your answer. {specific_feedback}"
            else:
                feedback_summary = "I've analyzed your answer and provided detailed feedback."

            # Update interview context
            while len(interview_context["answers"]) <= question_index:
                interview_context["answers"].append("")
            interview_context["answers"][question_index] = transcription

            while len(interview_context["feedback"]) <= question_index:
                interview_context["feedback"].append("")
            interview_context["feedback"][question_index] = analysis

            while len(interview_context["follow_ups"]) <= question_index:
                interview_context["follow_ups"].append([])
            interview_context["follow_ups"][question_index] = follow_up_questions

            # Save feedback to file
            feedback_file = os.path.join(coach.temp_dir, f"feedback_q{question_index+1}_{timestamp}.txt")
            with open(feedback_file, 'w') as f:
                f.write(f"Question: {question}\n\n")
                f.write(f"Your Answer: {transcription}\n\n")
                f.write(f"Feedback:\n{analysis}")

            return jsonify({
                "transcription": transcription,
                "feedback": analysis,
                "feedback_summary": feedback_summary,
                "follow_up_questions": follow_up_questions
            })

    except Exception as e:
        print(f"Error processing answer: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if not coach:
            return jsonify({"error": "Interview coach not initialized"}), 500

        # Create a unique filename for this speech
        timestamp = int(time.time())
        temp_file = os.path.join(coach.temp_dir, f"speech_{timestamp}.mp3")

        # Generate the speech file
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_file)

        if not os.path.exists(temp_file):
            return jsonify({"error": "Failed to create audio file"}), 500

        # Read and encode audio file as base64
        with open(temp_file, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        # Clean up the temp file
        try:
            os.remove(temp_file)
        except:
            pass

        return jsonify({
            "success": True,
            "audio_data": encoded_audio,
            "mime_type": "audio/mp3"
        })

    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/start-face-monitoring', methods=['POST'])
def start_face_monitoring():
    """Note: Face monitoring won't work on server-side. Implement client-side instead."""
    return jsonify({
        "success": True,
        "message": "Face monitoring should be implemented client-side using JavaScript"
    })


@app.route('/api/stop-face-monitoring', methods=['POST'])
def stop_face_monitoring():
    """Note: Face monitoring won't work on server-side."""
    return jsonify({"success": True})


@app.route('/api/generate-summary', methods=['POST'])
def generate_summary():
    global interview_context
    try:
        data = request.json
        client_questions = data.get('questions', [])
        client_answers = data.get('answers', [])

        # Use interview context data
        questions = interview_context.get("questions", client_questions)
        answers = interview_context.get("answers", client_answers)
        feedbacks = interview_context.get("feedback", [])
        job_role = interview_context.get("job_role", "professional")
        interview_type = interview_context.get("interview_type", "general")
        position_level = interview_context.get("position_level", "mid-level")

        if not coach:
            return jsonify({"error": "Interview coach not initialized"}), 500

        # Generate comprehensive summary
        summary_prompt = f"""As an interview coach for a {position_level} {job_role} position, provide a comprehensive summary of this {interview_type} interview session.

        Interview Type: {interview_type}
        Position Level: {position_level}

        Questions asked:
        {chr(10).join(f"- {q}" for q in questions)}

        Based on the candidate's answers and the feedback provided, please summarize:
        1. Overall performance (strengths and areas for improvement)
        2. Key recommendations for the candidate specific to {interview_type} interviews
        3. Suggested preparation steps for future {interview_type} interviews for {position_level} {job_role} positions

        Keep the summary concise but actionable, highlighting the most important points.
        don't add "*" when generating summary"""

        summary_response = coach.model.generate_content(summary_prompt)
        summary = summary_response.text.strip()

        # Generate condensed summary for TTS
        condensed_summary_prompt = f"""Based on this interview summary, create a brief (3-4 sentences) encouraging verbal conclusion
        to read to the candidate. Be specific but upbeat, focusing on key strengths and one area to work on. don't add "*" when generating summary:

        {summary}"""

        condensed_response = coach.model.generate_content(condensed_summary_prompt)
        condensed_summary = condensed_response.text.strip()

        # Save summary to file
        timestamp = int(time.time())
        summary_file = os.path.join(coach.temp_dir, f"interview_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Interview Summary for {position_level.capitalize()} {job_role} Position - {interview_type.capitalize()} Interview\n\n")
            f.write(summary)
            f.write("\n\nDetailed Question-Answer Records:\n\n")

            for q_idx, question in enumerate(questions):
                f.write(f"Question {q_idx + 1}: {question}\n\n")
                
                if q_idx < len(answers):
                    f.write(f"Answer: {answers[q_idx]}\n\n")
                
                if q_idx < len(feedbacks):
                    f.write(f"Feedback: {feedbacks[q_idx]}\n\n")
                
                if q_idx < len(interview_context.get("follow_ups", [])):
                    f.write("Follow-up questions:\n")
                    for follow_up in interview_context["follow_ups"][q_idx]:
                        f.write(f"- {follow_up}\n")
                    f.write("\n")
                
                f.write("-" * 50 + "\n\n")

        return jsonify({
            "summary": summary,
            "condensed_summary": condensed_summary
        })

    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/cleanup-question', methods=['POST'])
def cleanup_question():
    """Clean up temporary files for a specific question"""
    return jsonify({"success": True})


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean up all temporary files"""
    try:
        if coach:
            coach.cleanup_audio_files()
        
        global face_monitoring_active
        face_monitoring_active = False

        return jsonify({"success": True})

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy", "service": "AI Interview Coach"}), 200


# For local development only
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    print("Starting AI Interview Coach Web Application...")
    if coach:
        print(f"Using temporary directory: {coach.temp_dir}")

    # Run the Flask app (development only)
    app.run(debug=True, port=4000)
