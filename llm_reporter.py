"""
LLM-based Report Generation
Generates natural language analysis of anomaly detection metrics
"""
import os
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta

class LLMReporter:
    """
    Generates human-readable reports from detection metrics using LLM
    Supports OpenAI, Google Gemini, and local models (Ollama)
    """
    
    def __init__(self, provider: str = "groq", model: str = "openai/gpt-oss-120b"):
        """
        Args:
            provider: "openai", "gemini", "groq", or "ollama"
            model: Model name (e.g., "gpt-3.5-turbo", "gemini-2.5-flash", "mixtral-8x7b-32768", or "neural-chat")
        """
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        self.api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else \
                       os.getenv("GOOGLE_API_KEY") if provider == "gemini" else \
                       os.getenv("GROQ_API_KEY") if provider == "groq" else None
        
        # Caching to avoid hitting rate limits
        self.last_report_time = None
        self.last_report_cache = None
        self.cache_duration = 60  # Cache for 60 seconds
        
        if self.provider == "openai" and not self.api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set. LLM reports will not work.")
        elif self.provider == "gemini" and not self.api_key:
            print("⚠️  Warning: GOOGLE_API_KEY not set. LLM reports will not work.")
        elif self.provider == "groq" and not self.api_key:
            print("⚠️  Warning: GROQ_API_KEY not set. LLM reports will not work.")
    
    def _get_default_model(self) -> str:
        """Get default model for provider"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "gemini":
            return "gemini-pro"  # Available models: "gemini-pro", "gemini-1.5-flash"
        elif self.provider == "groq":
            return "mixtral-8x7b-32768"  # Fast & free. Other options: "llama2-70b-4096", "llama-2-13b-chat"
        elif self.provider == "ollama":
            return "neural-chat"  # Download with: ollama pull neural-chat
        return "gpt-3.5-turbo"
    
    def generate_report(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Generate a comprehensive report from anomaly metrics
        Uses caching to avoid hitting rate limits
        
        Args:
            metrics: Dictionary with anomaly metrics
        
        Returns:
            Natural language report or cached report
        """
        # Check cache (avoid hitting API too frequently)
        if self.last_report_cache and self.last_report_time:
            elapsed = (datetime.now() - self.last_report_time).total_seconds()
            if elapsed < self.cache_duration:
                print(f"📦 Using cached report (expires in {self.cache_duration - elapsed:.0f}s)")
                return self.last_report_cache
        
        if not self.api_key and self.provider in ["openai", "gemini"]:
            return self._fallback_report(metrics)
        
        prompt = self._build_prompt(metrics)
        report = None
        
        if self.provider == "openai":
            report = self._query_openai(prompt)
        elif self.provider == "gemini":
            report = self._query_gemini(prompt)
        elif self.provider == "groq":
            report = self._query_groq(prompt)
        elif self.provider == "ollama":
            report = self._query_ollama(prompt)
        
        # If LLM fails, use fallback
        if not report:
            report = self._fallback_report(metrics)
        
        # Cache the result
        self.last_report_cache = report
        self.last_report_time = datetime.now()
        
        return report
    
    def _build_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build detailed prompt for LLM"""
        anomalies_summary = "\n".join(
            [f"  - {k}: {v} occurrences" for k, v in metrics.get("anomaly_types", {}).items()]
        )
        
        top_tracks = "\n".join(
            [f"  - Track {t['track_id']}: {t['event_count']} events (confidence: {t['avg_confidence']:.2f})" 
             for t in metrics.get("top_tracks", [])[:5]]
        )
        
        prompt = f"""
Analyze the following crowd anomaly detection report and provide a professional executive summary with:
1. Overall safety assessment (Safe/Moderate Concern/High Risk)
2. Key findings and patterns
3. Top anomalies and affected persons
4. Recommendations for action
5. Time-based patterns if notable

METRICS DATA:
- Total Events Detected: {metrics.get('total_events', 0)}
- Total Alerts Generated: {metrics.get('total_alerts', 0)}
- Affected Individuals: {metrics.get('affected_tracks', 0)}
- Average Anomaly Confidence: {metrics.get('avg_confidence', 0):.2f} / 1.0
- Crowd Density: {metrics.get('crowd_density', 'Unknown')}

ANOMALY BREAKDOWN:
{anomalies_summary or '  - No anomalies detected'}

TOP PERSONS OF INTEREST:
{top_tracks or '  - No notable tracks'}

Processing Duration: {metrics.get('processing_time', 'N/A')} seconds
Analysis Timestamp: {metrics.get('timestamp', 'N/A')}

Please provide a concise but comprehensive report (200-400 words) suitable for security personnel or management review.
"""
        return prompt.strip()
    
    def _query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI API"""
        try:
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional security analyst providing crowd safety assessments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Google Gemini API with error handling"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            # Try to get available models if using default
            try:
                available_models = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                if available_models:
                    model_name = self.model if self.model in available_models else available_models[0]
                    print(f"Using Gemini model: {model_name}")
                else:
                    model_name = self.model
            except:
                model_name = self.model
            
            model = genai.GenerativeModel(model_name)
            
            # Add system context to prompt
            full_prompt = f"You are a professional security analyst providing crowd safety assessments.\n\n{prompt}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"⚠️  Gemini quota exceeded. Using template report instead.")
                print(f"💡 Tip: Check your API quota at https://ai.google.dev/rate-limits")
            else:
                print(f"Gemini API error: {e}")
            return None
    
    def _query_groq(self, prompt: str) -> Optional[str]:
        """Query Groq API (fast, free tier available)"""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            
            system_message = "You are a professional security analyst providing crowd safety assessments."
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            print(f"✓ Groq report generated using {self.model}")
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "quota" in error_str.lower():
                print(f"⚠️  Groq rate limit hit. Using template report instead.")
                print(f"💡 Tip: Groq has generous free tier. Check https://console.groq.com")
            else:
                print(f"Groq API error: {e}")
            return None
    
    def _query_ollama(self, prompt: str) -> Optional[str]:
        """Query local Ollama model"""
        try:
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            return None
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def _fallback_report(self, metrics: Dict[str, Any]) -> str:
        """Generate template-based report when LLM unavailable"""
        total_events = metrics.get("total_events", 0)
        total_alerts = metrics.get("total_alerts", 0)
        affected_tracks = metrics.get("affected_tracks", 0)
        avg_conf = metrics.get("avg_confidence", 0)
        
        # Determine risk level
        if total_alerts == 0:
            risk_level = "SAFE"
            assessment = "No significant anomalies detected during analysis period."
        elif avg_conf > 0.85:
            risk_level = "HIGH RISK"
            assessment = f"Multiple high-confidence anomalies detected ({total_alerts} alerts). Immediate review recommended."
        elif avg_conf > 0.75:
            risk_level = "MODERATE CONCERN"
            assessment = f"{total_alerts} alerts generated. Monitor situation closely."
        else:
            risk_level = "LOW RISK"
            assessment = f"Low-confidence anomalies detected ({total_alerts} alerts). Standard monitoring continues."
        
        anomalies = metrics.get("anomaly_types", {})
        top_anomaly = max(anomalies.items(), key=lambda x: x[1])[0] if anomalies else "None"
        
        report = f"""
CROWD ANOMALY DETECTION REPORT
═════════════════════════════════════════

SAFETY ASSESSMENT: {risk_level}
{assessment}

KEY METRICS:
• Total Events: {total_events}
• Total Alerts: {total_alerts}
• Affected Individuals: {affected_tracks}
• Average Confidence Score: {avg_conf:.2f}/1.0

TOP ANOMALY DETECTED: {top_anomaly}

RECOMMENDATIONS:
"""
        if risk_level == "HIGH RISK":
            report += "- Immediate security review required\n"
            report += "- Check top persons of interest (see dashboard)\n"
            report += "- Investigate alert hotspots on timeline\n"
        elif risk_level == "MODERATE CONCERN":
            report += "- Continue monitoring affected areas\n"
            report += "- Review top anomalies for patterns\n"
        else:
            report += "- Maintain standard monitoring protocols\n"
        
        report += f"\nGenerated: {metrics.get('timestamp', 'N/A')}"
        return report
