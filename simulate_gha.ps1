<#
.SYNOPSIS
Simulates the GitHub Actions deployment sequence locally for testing.

.DESCRIPTION
This script replicates the exact behavior of the deploy_pipeline.yml workflow:
1. Starts the Docker Compose stack.
2. Waits for health checks to pass (Airflow UI + MLflow).
3. Triggers the logistics_pipeline_dag.
4. Polls the DAG execution state until success or failure.
5. Extracts MLflow artifacts.
6. Gracefully shuts down the services.
#>

$ErrorActionPreference = 'Stop'

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Simulating GitHub Actions Pipeline Run " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Start Docker Services
Write-Host "`n[1/6] Starting Docker Services..." -ForegroundColor Yellow
docker-compose up --build -d

# Step 2: Wait for Health Checks
Write-Host "`n[2/6] Waiting for Airflow and MLflow to become healthy (timeout 3 mins)..." -ForegroundColor Yellow
$Timeout = 180
$StopWatch = [System.Diagnostics.Stopwatch]::StartNew()
$Healthy = $false

while ($StopWatch.Elapsed.TotalSeconds -lt $Timeout) {
    try {
        $AirflowResponse = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -ErrorAction SilentlyContinue
        $MlflowResponse = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -ErrorAction SilentlyContinue
        
        if ($AirflowResponse.StatusCode -eq 200 -and $MlflowResponse.StatusCode -eq 200) {
            $Healthy = $true
            Write-Host "Services are healthy!" -ForegroundColor Green
            break
        }
    } catch {
        # Catch network errors while the service boots up
    }
    Start-Sleep -Seconds 5
}

if (-not $Healthy) {
    Write-Error "Timeout waiting for services to become healthy."
    exit 1
}

# Step 3: Trigger Airflow DAG
Write-Host "`n[3/6] Triggering logistics_pipeline_dag..." -ForegroundColor Yellow
docker-compose exec -T airflow-webserver airflow dags trigger logistics_pipeline_dag

# Step 4: Poll DAG State
Write-Host "`n[4/6] Waiting for DAG to complete (timeout 15 mins)..." -ForegroundColor Yellow
$DagTimeout = 900
$StopWatch.Restart()
$DagSuccess = $false

while ($StopWatch.Elapsed.TotalSeconds -lt $DagTimeout) {
    # We grab the last line of the state output
    try {
        $StateOutput = docker-compose exec -T airflow-webserver airflow dags state logistics_pipeline_dag 2>$null
        $StateLines = $StateOutput -split "`r`n|`n" | Where-Object { $_.Trim() -ne "" }
        $State = $StateLines[-1]
        
        Write-Host "Current DAG state: $State"
        
        if ($State -match "success") {
            $DagSuccess = $true
            Write-Host "DAG execution SUCCEEDED!" -ForegroundColor Green
            break
        } elseif ($State -match "failed") {
            Write-Error "DAG execution FAILED!"
            break
        }
    } catch {
        Write-Warning "Failed to query DAG state. Retrying..."
    }
    
    Start-Sleep -Seconds 15
}

# Step 5: Extract Artifacts
if ($DagSuccess) {
    Write-Host "`n[5/6] Extracting MLflow Artifacts..." -ForegroundColor Yellow
    $MlflowCID = (docker-compose ps -q mlflow).Trim()
    if ($MlflowCID) {
        Write-Host "Copying data from container volume..."
        docker cp "${MlflowCID}:/mlflow" ./mlflow_export
        Write-Host "MLflow artifacts saved to ./mlflow_export." -ForegroundColor Green
        Write-Host "Model files are saved to data/models/." -ForegroundColor Green
    } else {
        Write-Warning "Could not find MLflow container to extract data."
    }
}

# Step 6: Teardown
Write-Host "`n[6/6] Tearing down services..." -ForegroundColor Yellow
docker-compose down -v
Write-Host "Teardown complete." -ForegroundColor Green

if ($DagSuccess) {
    Write-Host "`n[SUCCESS] GitHub Actions simulation completed successfully!" -ForegroundColor Cyan
} else {
    Write-Host "`n[FAILED] GitHub Actions simulation failed." -ForegroundColor Red
}
