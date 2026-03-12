pipeline {
    agent any

    stages {

        stage('Checkout Code') {
            steps {
                git 'https://github.com/YOUR_USERNAME/company-intelligence-agent.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                bat 'pytest tests'
            }
        }

        stage('Run Agent Pipeline') {
            steps {
                bat 'python main.py "wipro"'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t company-agent .'
            }
        }

    }
}