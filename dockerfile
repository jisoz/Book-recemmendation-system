FROM pytorch/pytorch

 RUN apt-get update && \
     apt-get install -y gcc python3-dev build-essential && \
         pip install --no-cache-dir numpy==1.23.5 pandas==1.5.3 gunicorn && \
             rm -rf /var/lib/apt/lists/*

             # Set the working directory in the container
             WORKDIR /app

             # Copy the current directory contents into the container at /app
             COPY . .

             # Install additional Python dependencies
             RUN pip install --no-cache-dir -r requirements.txt

             # Expose port 5000 to the outside world
             EXPOSE 5001

             # Command to run the Flask app with Gunicorn for production
             CMD ["gunicorn", "-b", "0.0.0.0:5001", "app:app"]