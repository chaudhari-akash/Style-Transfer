pipeline {
    agent any
    triggers {
        githubPush()
    }
    environment {
        DOCKER_FRONTEND_NAME = 'frontend'
        DOCKER_BACKEND_NAME = 'backend'
        DOCKER_MODELIMPORTER_NAME = 'model-importer'
        GITHUB_REPO_URL = 'https://github.com/chaudhari-akash/Style-Transfer.git'
    }
    stages {
        stage('Git Clone') {
            steps {
                script {
                    git branch: 'main', url: "${GITHUB_REPO_URL}"
                }
            }
        }
        stage('Build Frontend Image') {
            steps {
                script {
                    docker.build("${DOCKER_FRONTEND_NAME}", './frontend')
                }
            }
        }
        stage('Build Backend Image') {
            steps {
                script {
                    docker.build("${DOCKER_BACKEND_NAME}", './backend')
                }
            }
        }
        stage('Build ModelImporter Image') {
            steps {
                script {
                    docker.build("${DOCKER_MODELIMPORTER_NAME}", './mlflow-server')
                }
            }
        }
        stage('Push Frontend Image') {
            steps {
                script {
                    docker.withRegistry('', 'DockerHubCred') {
                        sh 'docker tag ${DOCKER_FRONTEND_NAME}:latest chaudhariakash/${DOCKER_FRONTEND_NAME}:latest'
                        sh 'docker push chaudhariakash/${DOCKER_FRONTEND_NAME}:latest'
                    }
                }
            }
        }
        stage('Push Backend Image') {
            steps {
                script {
                    docker.withRegistry('', 'DockerHubCred') {
                        sh 'docker tag ${DOCKER_BACKEND_NAME}:latest chaudhariakash/${DOCKER_BACKEND_NAME}:latest'
                        sh 'docker push chaudhariakash/${DOCKER_BACKEND_NAME}:latest'
                    }
                }
            }
        }
        stage('Push ModelImporter Image') {
            steps {
                script {
                    docker.withRegistry('', 'DockerHubCred') {
                        sh 'docker tag ${DOCKER_MODELIMPORTER_NAME}:latest chaudhariakash/${DOCKER_MODELIMPORTER_NAME}:latest'
                        sh 'docker push chaudhariakash/${DOCKER_MODELIMPORTER_NAME}:latest'
                    }
                }
            }
        }
        stage('Deploy to Kubernetes') {
            steps {
                withCredentials([file(credentialsId: 'MINIKUBE_KUBECONFIG', variable: 'KUBECONFIG')]){
                    ansiblePlaybook(
                    playbook: './ansible/playbook.yml',
                    inventory: './ansible/inventory.ini',
                    colorized: true
                )
                }
            }
        }
    }
    post {
        success {
            mail to: 'Chaudhari.Akash@iiitb.ac.in',
                subject: "Application Deployment SUCCESS: Build ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "The build was successful!"
        }
        failure {
            mail to: 'Chaudhari.Akash@iiitb.ac.in',
                subject: "Application Deployment FAILURE: Build ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "The build failed."
        }
        always {
            cleanWs()
        }
    }
}
