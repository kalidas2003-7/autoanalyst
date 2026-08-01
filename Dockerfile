FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs sessions

EXPOSE 8501

# ANTHROPIC_API_KEY can be passed at runtime to enable the LLM planner:
#   docker run -e ANTHROPIC_API_KEY=sk-... -p 8501:8501 autoanalyst
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "chat_app.py", "--server.port=8501"]
