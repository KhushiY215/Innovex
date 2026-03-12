pipeline {
    agent any

    parameters {
        string(name: 'COMPANY_NAME', defaultValue: 'apple', description: 'Company to analyze')
        string(name: 'MAX_ITERATIONS', defaultValue: '3', description: 'Max agent loop iterations')
    }

    environment {
        PYTHON = "C:\\Users\\Khushi Yadav\\AppData\\Local\\Programs\\Python\\Python313\\python.exe"
    }

    stages {

        stage('Install Dependencies') {
            steps {
                echo "Installing Python dependencies..."
                bat "\"%PYTHON%\" -m pip install --upgrade pip"
                bat "\"%PYTHON%\" -m pip install -r requirements.txt"
            }
        }

        stage('Run Validation Tests') {
            steps {
                echo "Running pytest..."
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