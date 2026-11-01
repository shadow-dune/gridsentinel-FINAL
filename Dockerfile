FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p .data && chmod -R a+rwX .data
ENV PORT=7860
EXPOSE 7860
CMD ["bash", "start.sh"]
