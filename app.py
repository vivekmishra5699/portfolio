import json
import os
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class RAGSystem:
    def __init__(self, json_file_path, model_name=None):
        """
        Initialize RAG system with semantic understanding
        """
        self.json_file_path = json_file_path
        
        # Load sentence transformer for embeddings (this does the smart matching)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load your personal data
        self.data = self.load_json_data()
        self.documents = self.prepare_documents()
        self.embeddings = self.create_embeddings()
        
        # No need for heavy LLM - we'll use smart semantic matching
        print("RAG System initialized with semantic understanding")

    def load_json_data(self):
        """Load JSON data from file"""
        try:
            import json
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"JSON file not found: {self.json_file_path}")
            return self.create_sample_data()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample JSON data structure"""
        sample_data = {
            "personal_info": {
                "name": "Your Name",
                "profession": "Software Developer",
                "location": "Your City",
                "bio": "I'm a passionate developer with expertise in AI and web development."
            },
            "experiences": [
                {
                    "company": "Tech Corp",
                    "role": "Senior Developer",
                    "duration": "2020-2023",
                    "description": "Developed AI-powered applications and led a team of 5 developers."
                }
            ],
            "skills": ["Python", "Machine Learning", "Flask", "JavaScript", "RAG Systems"],
            "projects": [
                {
                    "name": "AI Chatbot",
                    "description": "Built a conversational AI using transformers and RAG",
                    "technologies": ["Python", "Transformers", "Flask"]
                }
            ],
            "interests": ["AI Research", "Open Source", "Tech Innovation"],
            "education": {
                "degree": "Computer Science",
                "university": "Tech University",
                "year": "2019"
            }
        }
        
        # Save sample data
        with open(self.json_file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        return sample_data
    
    def prepare_documents(self):
        """Convert JSON data into searchable text documents"""
        documents = []
        
        # Personal info - updated key name
        if 'personal' in self.data:
            info = self.data['personal']
            doc = f"Personal Info: Name is {info.get('name', '')}, profession is {info.get('profession', '')}, located in {info.get('location', '')}. Bio: {info.get('brief_bio', '')}"
            documents.append(doc)
            
            # Add different introduction lengths
            if info.get('introduction_detailed'):
                documents.append(f"Detailed Introduction: {info['introduction_detailed']}")
            if info.get('introduction_medium'):
                documents.append(f"Medium Introduction: {info['introduction_medium']}")
            if info.get('introduction_short'):
                documents.append(f"Short Introduction: {info['introduction_short']}")
        
        # Work experience
        if 'work_experience' in self.data:
            for exp in self.data['work_experience']:
                doc = f"Work Experience: Worked at {exp.get('company', '')} as {exp.get('role', '')} from {exp.get('duration', '')}. {exp.get('description', '')}"
                documents.append(doc)
        
        # Skills - handle nested structure
        if 'skills' in self.data:
            for category in self.data['skills']:
                if isinstance(category, dict) and 'skills' in category:
                    category_name = category.get('category', 'General')
                    skills_in_category = []
                    
                    for skill in category['skills']:
                        if isinstance(skill, dict):
                            skill_name = skill.get('name', '')
                            proficiency = skill.get('proficiency', '')
                            years = skill.get('years', '')
                            details = skill.get('details', '')
                            
                            skill_text = skill_name
                            if proficiency:
                                skill_text += f" ({proficiency})"
                            if years:
                                skill_text += f" - {years} years"
                            if details:
                                skill_text += f" - {details}"
                            
                            skills_in_category.append(skill_text)
                    
                    if skills_in_category:
                        doc = f"Skills in {category_name}: {', '.join(skills_in_category)}"
                        documents.append(doc)
        
        # Projects
        if 'projects' in self.data:
            for proj in self.data['projects']:
                technologies = []
                if 'technologies' in proj:
                    for tech in proj['technologies']:
                        if isinstance(tech, str):
                            technologies.append(tech)
                        elif isinstance(tech, dict):
                            tech_name = tech.get('name', '') or tech.get('title', '')
                            if tech_name:
                                technologies.append(tech_name)
                
                project_doc = f"Project: {proj.get('name', '')} ({proj.get('duration', '')}) - {proj.get('description', '')}. Role: {proj.get('role', '')}. Technologies used: {', '.join(technologies)}"
                
                # Add challenges and solutions if available
                if proj.get('challenges'):
                    project_doc += f" Challenges: {proj['challenges']}"
                if proj.get('solutions'):
                    project_doc += f" Solutions: {proj['solutions']}"
                if proj.get('outcomes'):
                    project_doc += f" Outcomes: {proj['outcomes']}"
                
                documents.append(project_doc)
                
                # Add highlights as separate documents
                if proj.get('highlights'):
                    for highlight in proj['highlights']:
                        highlight_doc = f"Project Highlight for {proj.get('name', '')}: {highlight}"
                        documents.append(highlight_doc)
        
        # Interests
        if 'interests' in self.data:
            interests_list = []
            for interest in self.data['interests']:
                if isinstance(interest, str):
                    interests_list.append(interest)
                elif isinstance(interest, dict):
                    interest_name = interest.get('name', '') or interest.get('title', '')
                    if interest_name:
                        interests_list.append(interest_name)
            
            if interests_list:
                doc = f"Interests: {', '.join(interests_list)}"
                documents.append(doc)
        
        # Education
        if 'education' in self.data:
            for edu in self.data['education']:
                if isinstance(edu, dict):
                    doc = f"Education: {edu.get('degree', '')} from {edu.get('institution', '')} in {edu.get('year', '')} located in {edu.get('location', '')}. {edu.get('highlights', '')}"
                    documents.append(doc)
                    
                    # Add relevant courses
                    if edu.get('relevant_courses'):
                        courses_doc = f"Relevant Courses from {edu.get('institution', '')}: {', '.join(edu['relevant_courses'])}"
                        documents.append(courses_doc)
        
        # Additional Q&A pairs
        if 'additional_questions' in self.data:
            for qa in self.data['additional_questions']:
                qa_doc = f"Q: {qa.get('question', '')} A: {qa.get('answer', '')}"
                documents.append(qa_doc)
        
        return documents
    
    def create_embeddings(self):
        """Create embeddings for all documents"""
        return self.embedder.encode(self.documents)
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve most relevant documents for a query"""
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_docs = [self.documents[i] for i in top_indices]
        
        return relevant_docs, similarities[top_indices]
    
    def analyze_query_intent(self, query):
        """Enhanced query intent analysis"""
        query_lower = query.lower()
        
        # Define intent patterns with more specific matching
        intents = {
            'skills_technical': ['technical skills', 'programming skills', 'coding skills', 'technologies', 'tech stack', 'programming languages', 'what technologies', 'technical abilities'],
            'skills_general': ['skills', 'abilities', 'what can you do', 'your skills', 'skillset'],
            'projects': ['projects', 'work', 'built', 'developed', 'created', 'portfolio', 'worked on'],
            'project_names': ['project names', 'which projects', 'what projects', 'list projects'],
            'education': ['education', 'study', 'university', 'college', 'degree', 'academic'],
            'personal': ['who are you', 'about you', 'introduce', 'name', 'background', 'tell me about yourself', 'about yourself'],
            'experience': ['experience', 'worked', 'job', 'career', 'professional'],
            'interests': ['interests', 'hobbies', 'passionate about', 'enjoy'],
            'contact': ['contact', 'email', 'phone', 'reach', 'get in touch'],
            'goals': ['goals', 'objectives', 'career goals', 'long-term', 'aspirations', 'future plans'],
            'specific_project': [],
            'general': []
        }
        
        # Enhanced technical skills detection
        if any(phrase in query_lower for phrase in ['technical skills', 'tech skills', 'programming skills', 'technologies you know', 'technical abilities']):
            return 'skills_technical', {}
        
        # Enhanced general skills detection (but not if specifically asking for technical)
        if (any(phrase in query_lower for phrase in ['skills', 'abilities', 'what can you do']) and 
            'technical' not in query_lower):
            return 'skills_general', {}
        
        # Enhanced projects detection
        if any(phrase in query_lower for phrase in ['projects', 'worked on', 'built', 'developed', 'portfolio']):
            return 'projects', {}
        
        # Check for specific project names in the query
        project_names = []
        if 'projects' in self.data:
            for proj in self.data['projects']:
                name = proj.get('name', '').lower()
                if name and name in query_lower:
                    project_names.append(proj.get('name', ''))
        
        if project_names:
            return 'specific_project', {'project_names': project_names}
        
        # Match other intent patterns
        for intent, patterns in intents.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent, {}
        
        return 'general', {}
    
    def get_targeted_documents(self, intent, intent_data=None):
        """Enhanced document targeting"""
        targeted_docs = []
        
        if intent == 'skills_technical':
            # Get ONLY technical skills - be very specific
            for doc in self.documents:
                if (doc.startswith("Skills in") and 
                    any(tech in doc.lower() for tech in ['programming', 'ai/ml', 'web development', 'python', 'javascript', 'tensorflow', 'pytorch', 'flask', 'git', 'algorithms', 'data structures']) and
                    'languages:' not in doc.lower() and
                    'soft skills' not in doc.lower()):
                    targeted_docs.append(doc)
            
            # Also add Q&A specifically about technical skills
            for doc in self.documents:
                if (doc.startswith("Q: ") and 
                    any(phrase in doc.lower() for phrase in ['technical skills', 'programming skills', 'technologies']) and
                    'non-technical' not in doc.lower()):
                    targeted_docs.append(doc)
        
        elif intent == 'skills_general':
            # Get ALL skills documents
            for doc in self.documents:
                if doc.startswith("Skills in"):
                    targeted_docs.append(doc)
            
            # Add general skills Q&A but exclude very specific ones
            for doc in self.documents:
                if (doc.startswith("Q: ") and 
                    any(word in doc.lower() for word in ['skills', 'abilities']) and
                    not any(phrase in doc.lower() for phrase in ['most valuable', 'communicate', 'methodology', 'non-technical'])):
                    targeted_docs.append(doc)
        
        elif intent == 'projects' or intent == 'project_names':
            # Get project information
            for doc in self.documents:
                if doc.startswith("Project:"):
                    targeted_docs.append(doc)
            
            # Add project-related Q&A
            for doc in self.documents:
                if (doc.startswith("Q: ") and 
                    any(word in doc.lower() for word in ['project', 'built', 'developed', 'worked on'])):
                    targeted_docs.append(doc)
        
        elif intent == 'specific_project' and intent_data:
            # Get specific project information
            project_names = intent_data.get('project_names', [])
            for doc in self.documents:
                if doc.startswith("Project:") and any(name in doc for name in project_names):
                    targeted_docs.append(doc)
        
        elif intent == 'education':
            # Get education information
            for doc in self.documents:
                if doc.startswith("Education:") or "Relevant Courses" in doc:
                    targeted_docs.append(doc)
        
        elif intent == 'personal':
            # Get personal information
            for doc in self.documents:
                if any(start in doc for start in ["Personal Info:", "Detailed Introduction:", "Medium Introduction:", "Short Introduction:"]):
                    targeted_docs.append(doc)
        
        elif intent == 'experience':
            # Get work experience
            for doc in self.documents:
                if doc.startswith("Work Experience:"):
                    targeted_docs.append(doc)
        
        elif intent == 'interests':
            # Get interests
            for doc in self.documents:
                if doc.startswith("Interests:"):
                    targeted_docs.append(doc)
        
        elif intent == 'contact':
            # Get contact information from personal info
            for doc in self.documents:
                if "Personal Info:" in doc and any(contact in doc.lower() for contact in ['email', 'phone', '@']):
                    targeted_docs.append(doc)
        
        elif intent == 'goals':
            # Get goals and career objectives
            for doc in self.documents:
                if (doc.startswith("Q: ") and 
                    any(word in doc.lower() for word in ['goals', 'career', 'objective', 'aspirations', 'future'])):
                    targeted_docs.append(doc)
        
        else:  # general fallback
            return None
        
        return targeted_docs[:5] if targeted_docs else None

    def generate_response(self, query, max_length=150):
        """Generate response using semantic understanding"""
        
        print(f"Query: {query}")
        
        # Step 1: Use semantic similarity to find the most relevant documents
        relevant_docs, scores = self.retrieve_relevant_docs(query, top_k=8)
        print(f"Found {len(relevant_docs)} relevant documents")
        
        # Step 2: Smart answer extraction using semantic understanding
        smart_answer = self.extract_smart_answer(query, relevant_docs, scores)
        if smart_answer:
            return {
                'response': smart_answer,
                'relevant_docs': relevant_docs[:3],
                'confidence_scores': scores[:3].tolist() if hasattr(scores, 'tolist') else scores[:3],
                'method': 'smart_extraction'
            }
        
        # Step 3: Fallback to contextual response generation
        contextual_response = self.generate_contextual_response(query, relevant_docs)
        
        return {
            'response': contextual_response,
            'relevant_docs': relevant_docs[:3],
            'confidence_scores': scores[:3].tolist() if hasattr(scores, 'tolist') else scores[:3],
            'method': 'contextual_generation'
        }

    def construct_project_answer(self, query, project_docs):
        """Construct detailed project answer from project documents"""
        query_lower = query.lower()
        
        # Find the most relevant project
        target_project = None
        
        # Check for specific project names in query
        project_mappings = {
            'solar system': 'Three.js Solar System Simulation',
            'chess': 'Deep Learning AI Chess Bot',
            'emergency': 'Emergency System Project',
            'ruralcare': 'RuralCare_AI',
            'three.js': 'Three.js Solar System Simulation'
        }
        
        for keyword, project_name in project_mappings.items():
            if keyword in query_lower:
                # Find matching project docs
                matching_docs = [doc for doc in project_docs if project_name in doc]
                if matching_docs:
                    target_project = project_name
                    project_docs = matching_docs
                    break
        
        if not target_project and project_docs:
            # Use the first project if no specific one mentioned
            first_project_doc = project_docs[0]
            if "Project:" in first_project_doc:
                try:
                    target_project = first_project_doc.split(" (")[0].replace("Project: ", "")
                except IndexError:
                    target_project = first_project_doc.replace("Project: ", "").split(" - ")[0] if " - " in first_project_doc else "Unknown Project"
        
        # Construct comprehensive project answer
        main_info = []
        highlights = []
        technologies = []
        
        for doc in project_docs:
            try:
                if doc.startswith("Project:") and (target_project in doc if target_project else True):
                    # Extract main project information more safely
                    if " - " in doc:
                        parts = doc.split(" - ", 1)  # Split only on first occurrence
                        if len(parts) >= 2:
                            description = parts[1].split(".")[0]
                            main_info.append(description)
                
                # Extract technologies safely
                if "Technologies used:" in doc:
                    try:
                        tech_section = doc.split("Technologies used: ")[1]
                        # Split on common delimiters to find the end of technologies section
                        tech_end_markers = [" Challenges:", " Solutions:", " Outcomes:", " Role:"]
                        tech_text = tech_section
                        
                        for marker in tech_end_markers:
                            if marker in tech_text:
                                tech_text = tech_text.split(marker)[0]
                                break
                        
                        technologies.append(tech_text.strip())
                    except (IndexError, AttributeError):
                        pass
                
                # Extract challenges safely
                if "Challenges:" in doc:
                    try:
                        challenges_section = doc.split("Challenges: ")[1]
                        challenges = challenges_section.split(" Solutions:")[0] if " Solutions:" in challenges_section else challenges_section.split(" Outcomes:")[0] if " Outcomes:" in challenges_section else challenges_section
                        main_info.append(f"Key challenges included {challenges}")
                    except (IndexError, AttributeError):
                        pass
                
                # Extract solutions safely
                if "Solutions:" in doc:
                    try:
                        solutions_section = doc.split("Solutions: ")[1]
                        solutions = solutions_section.split(" Outcomes:")[0] if " Outcomes:" in solutions_section else solutions_section
                        main_info.append(f"I solved these by {solutions}")
                    except (IndexError, AttributeError):
                        pass
                
                # Extract outcomes safely
                if "Outcomes:" in doc:
                    try:
                        outcomes = doc.split("Outcomes: ")[1]
                        main_info.append(f"The result was {outcomes}")
                    except (IndexError, AttributeError):
                        pass
            
            except Exception as e:
                print(f"Error processing project doc: {e}")
                continue
                
            # Fixed the indentation issue here
            if doc.startswith("Project Highlight") and (target_project in doc if target_project else True):
                try:
                    # Extract highlight more safely
                    if ": " in doc:
                        parts = doc.split(": ")
                        if len(parts) >= 3:
                            highlight = parts[2]
                        elif len(parts) >= 2:
                            highlight = parts[1]
                        else:
                            highlight = doc.replace("Project Highlight", "").strip()
                        
                        highlights.append(highlight)
                except (IndexError, AttributeError):
                    pass
        
        # Build the response
        if not target_project:
            target_project = "this project"
        
        response_parts = []
        
        if main_info:
            response_parts.append(f"The {target_project} is {main_info[0]}")
            if len(main_info) > 1:
                response_parts.extend(main_info[1:])
        else:
            # Fallback description
            response_parts.append(f"The {target_project} is an interactive web application")
        
        if technologies:
            response_parts.append(f"Built using {technologies[0]}")
        
        if highlights:
            response_parts.append(f"Key features include: {', '.join(highlights[:3])}")
        
        # Join all parts into a coherent response
        if response_parts:
            return ". ".join(response_parts) + "."
        
        # Final fallback
        return f"I worked on the {target_project}, which involved creating an interactive application with modern web technologies."

    def extract_smart_answer(self, query, docs, scores):
        """Smart answer extraction using semantic understanding"""
        
        # Enhanced project-specific handling
        query_lower = query.lower()
        
        # Check if asking about a specific project
        project_keywords = ['project', 'simulation', 'chess', 'solar system', 'emergency', 'ruralcare', 'ai bot']
        asking_about_project = any(keyword in query_lower for keyword in project_keywords)
        
        if asking_about_project:
            # Prioritize project documents over Q&A for project queries
            project_docs = [doc for doc in docs if doc.startswith("Project:") or doc.startswith("Project Highlight")]
            if project_docs:
                try:
                    return self.construct_project_answer(query, project_docs)
                except Exception as e:
                    print(f"Error constructing project answer: {e}")
                    # Fallback to simple project description
                    for doc in project_docs:
                        if doc.startswith("Project:") and any(keyword in doc.lower() for keyword in ['solar system', 'three.js']):
                            # Simple extraction
                            if " - " in doc:
                                description = doc.split(" - ")[1].split(".")[0]
                                return f"The Three.js Solar System Simulation is {description}."
                            return "The Three.js Solar System Simulation is an interactive 3D solar system visualization built with JavaScript and Three.js."
        
        # Enhanced skills detection - check before Q&A matching
        technical_categories = ['programming', 'ai/ml', 'web development', 'tools', 'frameworks', 'libraries', 'databases', 'cloud']
        
        if any(f"{cat} skills" in query_lower for cat in technical_categories) or (
            any(word in query_lower for word in ['tools', 'programming', 'technical', 'technologies']) and 'skills' in query_lower
        ):
            print(f"Direct technical skills detection: {query}")
            return self.answer_about_skills(docs, query_lower, is_technical=True)
        
        # Enhanced Q&A matching with better intent detection
        for i, doc in enumerate(docs):
            if doc.startswith("Q: ") and "A: " in doc:
                try:
                    question_part = doc.split("A: ")[0].replace("Q: ", "")
                    answer_part = doc.split("A: ")[1]
                    
                    # Use semantic similarity AND keyword matching for better precision
                    question_embedding = self.embedder.encode([question_part])
                    query_embedding = self.embedder.encode([query])
                    similarity = cosine_similarity(query_embedding, question_embedding)[0][0]
                    
                    # Enhanced matching with intent-specific thresholds
                    question_lower = question_part.lower()
                    
                    # Skills-specific queries need higher precision and better matching
                    if any(skill_word in query_lower for skill_word in ['skills', 'abilities', 'technical', 'technologies', 'programming', 'tools']):
                        # Only match if the Q&A is actually about the same type of skills
                        if not any(skill_word in question_lower for skill_word in ['skills', 'abilities', 'technical', 'technologies', 'programming', 'tools']):
                            continue
                        
                        # Check for category match
                        query_categories = [cat for cat in technical_categories if cat in query_lower]
                        question_categories = [cat for cat in technical_categories if cat in question_lower]
                        
                        if query_categories and question_categories and not any(qcat in question_categories for qcat in query_categories):
                            continue
                        
                        # Increase threshold for skills queries
                        min_similarity = 0.7
                    else:
                        min_similarity = 0.4
                    
                    # Identity/about queries
                    if any(identity_word in query_lower for identity_word in ['about', 'yourself', 'who are you', 'describe']):
                        # Only match identity-related Q&As
                        if not any(identity_word in question_lower for identity_word in ['describe', 'yourself', 'who are you', 'about']):
                            continue
                        min_similarity = 0.5
                    
                    if similarity > min_similarity:
                        # Additional keyword validation to avoid wrong matches
                        
                        # Skip if asking for technical skills but got non-technical skills answer
                        if ('technical' in query_lower and 'non-technical' in question_lower):
                            continue
                            
                        # Skip if asking for general skills but got very specific questions
                        if (any(word in query_lower for word in ['skills', 'abilities']) and 
                            any(phrase in question_lower for phrase in ['most valuable', 'communicate', 'methodology', 'taught you'])):
                            continue
                        
                        # Skip methodology/approach answers when asking about specific projects
                        if (asking_about_project and 
                            any(phrase in question_lower for phrase in ['approach', 'methodology', 'breaking down', 'tackled', 'systematically'])):
                            continue
                        
                        # Skip if asking about specific project but getting general project methodology
                        if (any(project in query_lower for project in ['solar system', 'chess', 'emergency', 'ruralcare']) and
                            not any(project in question_lower for project in ['solar system', 'chess', 'emergency', 'ruralcare'])):
                            continue
                        
                        # Skip professional description for skills queries
                        if (any(skill_word in query_lower for skill_word in ['skills', 'abilities', 'technical', 'tools']) and 
                            'professionally' in question_lower):
                            continue
                        
                        print(f"Found semantic match: {similarity:.3f} - {question_part}")
                        return answer_part
                
                except Exception as e:
                    print(f"Error processing Q&A document: {e}")
                    continue
        
        # If no direct Q&A match, try to construct answer from relevant information
        try:
            return self.construct_smart_answer(query, docs, scores)
        except Exception as e:
            print(f"Error in construct_smart_answer: {e}")
            return "I'd be happy to tell you more about my skills. Could you please rephrase your question?"

    def construct_smart_answer(self, query, docs, scores):
        """Construct intelligent answers based on query intent and available information"""
        
        # Store current query for use in other methods
        self._current_query = query
        
        query_lower = query.lower()
        
        # Detect what the user is asking about using semantic understanding
        query_embedding = self.embedder.encode([query])
        
        # Enhanced semantic patterns with better technical detection
        question_patterns = {
            'identity': ['who are you', 'tell me about yourself', 'introduce yourself', 'your name', 'describe yourself', 'about you'],
            'skills_technical': [
                'technical skills', 'programming skills', 'technologies', 'tech stack', 'coding skills', 'technical abilities',
                'programming languages', 'frameworks', 'libraries', 'tools', 'software tools', 'development tools',
                'ai skills', 'ml skills', 'web development skills', 'database skills', 'cloud skills'
            ],
            'skills_general': ['skills', 'abilities', 'what can you do', 'your skills', 'skillset', 'soft skills'],
            'goals': ['goals', 'objectives', 'aspirations', 'future plans', 'career goals', 'what do you want'],
            'projects': ['projects', 'work', 'built', 'developed', 'portfolio', 'what have you made', 'simulation', 'chess bot', 'emergency system'],
            'education': ['education', 'study', 'university', 'degree', 'school', 'learning'],
            'experience': ['experience', 'worked', 'job', 'career', 'professional background'],
            'interests': ['interests', 'hobbies', 'passionate about', 'like to do', 'enjoy'],
            'contact': ['contact', 'reach', 'email', 'phone', 'get in touch'],
            'location': ['where', 'location', 'live', 'from', 'based'],
        }
        
        # Enhanced technical skills detection with specific category checking
        technical_categories = [
            'programming', 'ai/ml', 'web development', 'tools', 'frameworks', 'libraries', 
            'databases', 'cloud', 'devops', 'machine learning', 'deep learning', 'frontend', 
            'backend', 'javascript', 'python', 'tensorflow', 'pytorch', 'flask', 'react'
        ]
        
        # Check if asking about specific technical skill categories
        if any(f"{cat} skills" in query_lower for cat in technical_categories):
            print(f"Detected technical category query for: {query}")
            return self.answer_about_skills(docs, query_lower, is_technical=True)
        
        # Check if asking about specific technical skills by name
        if (any(word in query_lower for word in ['tools', 'programming', 'technical', 'technologies', 'framework', 'library']) 
            and 'skills' in query_lower):
            print(f"Detected technical skills query: {query}")
            return self.answer_about_skills(docs, query_lower, is_technical=True)
        
        # Enhanced project detection
        if any(keyword in query_lower for keyword in ['project', 'simulation', 'chess', 'solar system', 'emergency', 'built', 'developed', 'created']):
            return self.answer_about_projects(docs)
        
        # Find the best matching intent using semantic similarity
        best_intent = None
        best_similarity = 0
        
        for intent, patterns in question_patterns.items():
            for pattern in patterns:
                pattern_embedding = self.embedder.encode([pattern])
                similarity = cosine_similarity(query_embedding, pattern_embedding)[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_intent = intent
        
        print(f"Detected intent: {best_intent} (confidence: {best_similarity:.3f})")
        
        # Generate response based on detected intent and available documents
        if best_intent == 'identity':
            return self.answer_about_identity(docs)
        elif best_intent in ['skills_technical', 'skills_general']:
            # Determine if it's really technical based on context
            is_technical = ('technical' in query_lower or 
                           any(cat in query_lower for cat in technical_categories) or
                           best_intent == 'skills_technical')
            return self.answer_about_skills(docs, query_lower, is_technical=is_technical)
        elif best_intent == 'goals':
            return self.answer_about_goals(docs)
        elif best_intent == 'projects':
            return self.answer_about_projects(docs)
        elif best_intent == 'education':
            return self.answer_about_education(docs)
        elif best_intent == 'experience':
            return self.answer_about_experience(docs)
        elif best_intent == 'interests':
            return self.answer_about_interests(docs)
        elif best_intent == 'contact':
            return self.answer_about_contact(docs)
        elif best_intent == 'location':
            return self.answer_about_location(docs)
        else:
            return self.generate_contextual_response(query, docs)

    def answer_about_skills(self, docs, query_lower=None, is_technical=None):
        """Enhanced skills response with better category detection"""
        if not query_lower:
            query_lower = getattr(self, '_current_query', '').lower()
        
        # Auto-detect if technical if not specified
        if is_technical is None:
            technical_indicators = [
                'technical', 'programming', 'coding', 'technologies', 'tech', 'tools',
                'frameworks', 'libraries', 'languages', 'ai/ml', 'web development',
                'python', 'javascript', 'tensorflow', 'pytorch', 'flask', 'react'
            ]
            is_technical = any(indicator in query_lower for indicator in technical_indicators)
        
        # Check for specific skill category in query
        specific_category = None
        category_mappings = {
            'tools': ['Tools', 'Development Tools', 'Software Tools'],
            'programming': ['Programming', 'Programming Languages'],
            'ai': ['AI/ML', 'Machine Learning', 'Artificial Intelligence'],
            'web': ['Web Development', 'Frontend', 'Backend'],
            'languages': ['Languages', 'Programming Languages'],
            'frameworks': ['Frameworks', 'Web Frameworks'],
            'databases': ['Databases', 'Database'],
            'cloud': ['Cloud', 'Cloud Computing']
        }
        
        for key, categories in category_mappings.items():
            if key in query_lower:
                specific_category = categories
                break
        
        skills_info = []
        category_skills = {}
        
        # Parse skills documents
        for doc in docs:
            if doc.startswith("Skills in "):
                # Extract category and skills
                try:
                    category_part = doc.split("Skills in ")[1].split(":")[0]
                    skills_part = doc.split(": ")[1] if ": " in doc else ""
                    
                    # Store by category
                    category_skills[category_part] = skills_part
                    
                    # If asking for specific category, prioritize it
                    if specific_category and any(cat.lower() in category_part.lower() for cat in specific_category):
                        skills_info.insert(0, f"My {category_part} skills include: {skills_part}")
                    elif is_technical and any(tech_cat in category_part.lower() for tech_cat in ['programming', 'ai/ml', 'web development', 'tools']):
                        skills_info.append(f"In {category_part}: {skills_part}")
                    elif not is_technical:
                        skills_info.append(f"In {category_part}: {skills_part}")
                        
                except (IndexError, AttributeError) as e:
                    print(f"Error parsing skills doc: {e}")
                    continue
    
        # Construct response based on query specificity
            if specific_category and skills_info:
            # Specific category requested
             return skills_info[0] + "."
        
            elif is_technical and skills_info:
                # Technical skills requested
                technical_skills = [skill for skill in skills_info if any(tech in skill.lower() for tech in ['programming', 'ai/ml', 'web development', 'tools', 'frameworks'])]
                if technical_skills:
                    response = "My technical skills include: " + ". ".join(technical_skills[:3])
                    return response + "."
                else:
                    return "I have technical skills in programming, AI/ML, web development, and various development tools."
        
            elif skills_info:
            # General skills overview
                response = "My skills span multiple areas: " + ". ".join(skills_info[:3])
                return response + "."
        
        # Enhanced fallback based on query
        if 'tools' in query_lower:
            return "My development tools include Git, VS Code, Jupyter Notebook, and various IDEs for different programming languages."
        elif 'programming' in query_lower:
            return "My programming skills include Python, JavaScript, Java, and C++, with experience in various frameworks and libraries."
        elif 'ai' in query_lower or 'ml' in query_lower:
            return "My AI/ML skills include TensorFlow, PyTorch, scikit-learn, and experience with deep learning, computer vision, and natural language processing."
        elif 'web' in query_lower:
            return "My web development skills include Flask, JavaScript, HTML/CSS, Three.js, and experience with both frontend and backend development."
        
        # General fallback
        return "I have a diverse skill set including programming languages (Python, JavaScript, Java), AI/ML frameworks (TensorFlow, PyTorch), web technologies (Flask, Three.js), and various development tools."

    def answer_about_identity(self, docs):
        """Generate response about personal identity/introduction"""
        personal_info = []
        brief_bio = None
        
        for doc in docs:
            if "brief_bio:" in doc:
                brief_bio = doc.split("brief_bio: ")[1]
            elif "Name:" in doc or "Profession:" in doc or "Location:" in doc:
                info = doc.split(": ", 1)[1] if ": " in doc else doc
                personal_info.append(info)
            elif doc.startswith("Personal: "):
                info = doc.replace("Personal: ", "")
                personal_info.append(info)
        
        if brief_bio:
            return brief_bio
        elif personal_info:
            return "I'm " + ". ".join(personal_info[:3]) + "."
        else:
            return "I'm Vivek Mishra, a Computer Science student passionate about AI and web technologies. I enjoy building innovative projects and solving complex problems."

    def answer_about_goals(self, docs):
        """Generate response about goals and aspirations"""
        for doc in docs:
            if "career goal" in doc.lower() or "objective" in doc.lower():
                if "A: " in doc:
                    return doc.split("A: ")[1]
                return doc
        
        return "I'm passionate about developing intelligent systems and immersive digital experiences. I'm seeking opportunities to apply my skills in cutting-edge projects that challenge and expand my expertise in AI and web technologies."

    def answer_about_projects(self, docs):
        """Generate response about projects"""
        projects = []
        for doc in docs:
            if doc.startswith("Project: "):
                project_name = doc.split(" (")[0].replace("Project: ", "")
                if " - " in doc:
                    description = doc.split(" - ")[1].split(".")[0]
                    projects.append(f"{project_name}: {description}")
                else:
                    projects.append(project_name)
        
        if projects:
            return "I've worked on several projects including " + "; ".join(projects[:3]) + ". Each project taught me valuable skills in AI, web development, and problem-solving."
        
        return "I've worked on projects like a Deep Learning AI Chess Bot, Three.js Solar System Simulation, and Emergency System Project."

    def answer_about_education(self, docs):
        """Generate response about education"""
        for doc in docs:
            if doc.startswith("Education:"):
                return doc.replace("Education: ", "I studied ")
            elif "Computer Science" in doc and "student" in doc:
                return "I'm currently a Computer Science and Cloud Computing student at Loyola Academy in Hyderabad."
        
        return "I'm a Computer Science and Cloud Computing student at Loyola Academy in Hyderabad."

    def answer_about_experience(self, docs):
        """Generate response about work experience"""
        for doc in docs:
            if doc.startswith("Work Experience:"):
                return doc.replace("Work Experience: ", "")
        
        return "I'm currently a student focusing on building my skills through various projects and coursework in AI and web technologies."

    def answer_about_interests(self, docs):
        """Generate response about interests"""
        for doc in docs:
            if doc.startswith("Interests:"):
                interests = doc.replace("Interests: ", "")
                return f"I'm passionate about {interests}. I particularly enjoy working on projects that combine these interests."
        
        return "I'm passionate about Artificial Intelligence, Game Development, and Web Technologies. I love creating interactive experiences and intelligent systems."

    def answer_about_contact(self, docs):
        """Generate response about contact information"""
        email = phone = None
        for doc in docs:
            if "@" in doc:
                # Extract email
                parts = doc.split()
                for part in parts:
                    if "@" in part:
                        email = part.replace(",", "").replace(".", "")
                        break
            if "+91" in doc or "phone" in doc.lower():
                # Extract phone
                import re
                phone_match = re.search(r'\+91\s*\d{10}', doc)
                if phone_match:
                    phone = phone_match.group()
        
        response = "You can reach me "
        if email:
            response += f"at {email}"
        if phone:
            if email:
                response += f" or call me at {phone}"
            else:
                response += f"at {phone}"
        
        if email or phone:
            response += ". I'm always open to discussing new opportunities!"
            return response
        
        return "You can reach me at vivek29403@gmail.com or +91 8074743216. I'm always open to new opportunities!"

    def answer_about_location(self, docs):
        """Generate response about location"""
        for doc in docs:
            if "located in" in doc or "Hyderabad" in doc:
                return "I'm based in Hyderabad, India."
        
        return "I'm based in Hyderabad, India."

    def generate_contextual_response(self, query, docs):
        """Generate contextual response when no specific pattern matches"""
        
        # Extract the most relevant information from top documents
        relevant_info = []
        for doc in docs[:3]:
            if doc.startswith("Q: ") and "A: " in doc:
                answer = doc.split("A: ")[1]
                relevant_info.append(answer)
            elif any(doc.startswith(prefix) for prefix in ["Personal Info:", "Project:", "Skills in", "Education:", "Interests:"]):
                relevant_info.append(doc)
        
        if relevant_info:
            # Create a contextual response based on available information
            return f"Based on what I know, {relevant_info[0][:200]}..." if len(relevant_info[0]) > 200 else relevant_info[0]
        
        return "I'd be happy to tell you more about my background, projects, or skills. What specifically would you like to know?"

# Initialize RAG system
rag_system = RAGSystem('my_personal_data.json')

@app.route('/')
def home():
    """Portfolio home page"""
    return render_template('portfolio.html')

@app.route('/chat')
def chat_page():
    """Dedicated chat page"""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Chat API endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Generate response using RAG system
        result = rag_system.generate_response(message)
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'sources': result.get('relevant_docs', []),
            'method': result.get('method', 'unknown')
        })
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/projects')
def projects():
    """Projects showcase page"""
    return render_template('projects.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/api/portfolio-data')
def portfolio_data():
    """API endpoint for portfolio data"""
    return jsonify(rag_system.data)

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("Starting Portfolio Application...")
    app.run(debug=True, host='0.0.0.0', port=5000)