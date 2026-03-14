pipeline {
    agent any

    parameters {
        string(name: 'COMPANY_NAME', defaultValue: 'apple', description: 'Company to analyze')
        string(name: 'MAX_ITERATIONS', defaultValue: '3', description: 'Max agent loop iterations')
    }

    environment {

        PYTHON = "C:\\Users\\Khushi Yadav\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"

        PYTHONUTF8 = "1"
        PYTHONIOENCODING = "utf-8"

        NVIDIA_API_KEY = credentials('nvidia-key')
        GROQ_API_KEY = credentials('groq-key')
        CEREBRAS_API_KEY = credentials('cerebras-key')
        HF_TOKEN = credentials('hf-key')
        LANGCHAIN_API_KEY = credentials('langsmith-key')
        SUPABASE_KEY = credentials('supabase-key')

        LANGCHAIN_TRACING_V2 = "true"
        LANGCHAIN_PROJECT = "company-intelligence-agent"

        HF_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
        NVIDIA_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
        CEREBRAS_MODEL = "llama3.1-8b"
        GROQ_MODEL = "llama-3.3-70b-versatile"
        GROQ_JUDGE_MODEL = "qwen/qwen3-32b"

        OUTPUT_DIR = "outputs"
        LOG_LEVEL = "INFO"

        SUPABASE_URL = "https://ymgwsnkbbxfmogxhoxgm.supabase.co"
    }

    stages {

        stage('Install Dependencies') {
            steps {
                echo "Installing Python dependencies..."
                bat "\"%PYTHON%\" -m pip install --upgrade pip"
                bat "\"%PYTHON%\" -m pip install --no-cache-dir -r requirements.txt"
            }
        }

        stage('Run Tests') {
            steps {
                echo "Running validation tests..."
                bat "\"%PYTHON%\" -m pytest tests"
            }
        }

        stage('Run Agent Pipeline') {
            steps {
                echo "Running LangGraph pipeline..."
                bat "\"%PYTHON%\" main.py %COMPANY_NAME% --max-iterations %MAX_ITERATIONS%"
            }
        }

        stage('Build Docker Image') {
            when {
                expression { currentBuild.result == null || currentBuild.result == 'SUCCESS' }
            }
            steps {
                echo "Building Docker image..."
                bat "docker build -t company-agent ."
            }
        }
    }

    post {
        always {
            echo "Pipeline finished"
        }
        success {
            echo "Pipeline succeeded"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}