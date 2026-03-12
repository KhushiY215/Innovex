pipeline {
    agent any

    parameters {
        string(
            name: 'COMPANY_NAME',
            defaultValue: 'apple',
            description: 'Company to analyze'
        )

        string(
            name: 'MAX_ITERATIONS',
            defaultValue: '3',
            description: 'Max Agent loop iterations'
        )
    }

    environment {
        PYTHON = "python"
    }

    stages {

        stage('Checkout Repository') {
            steps {
                echo "Cloning repository..."
                git 'https://github.com/KhushiY215/Innovex.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo "Installing Python dependencies..."
                bat "%PYTHON% -m pip install --upgrade pip"
                bat "%PYTHON% -m pip install -r requirements.txt"
            }
        }

        stage('Run Validation Tests') {
            steps {
                echo "Running pytest validation..."
                bat "%PYTHON% -m pytest tests"
            }
        }

        stage('Run Agent Pipeline') {
            steps {
                echo "Running LangGraph company pipeline..."

                bat """
                %PYTHON% main.py "%COMPANY_NAME%" --max-iterations %MAX_ITERATIONS%
                """
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."
                bat "docker build -t company-intelligence-agent ."
            }
        }

    }

    post {

        success {
            echo "Pipeline completed successfully"
        }

        failure {
            echo "Pipeline failed"
        }

        always {
            echo "Pipeline finished"
        }

    }
}