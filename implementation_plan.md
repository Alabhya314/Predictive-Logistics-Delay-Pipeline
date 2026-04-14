# Zero-Cost Deployed Solution Plan

The goal is to provide a "zero cost deployed solution" for the End-to-End Logistics Pipeline, which currently uses a heavy set of technologies (Airflow, MLFlow, Postgres). 

## User Review Required

> [!WARNING]
> This pipeline relies on Airflow and MLflow which require at least ~2GB-3GB of RAM memory to run. Because of this, it is impossible to host this 24/7 permanently on traditional free-tier environments (like Render or Fly.io), as those platforms have strict limits that the `docker-compose` stack will immediately hit and crash.

**The Solution:** We will deploy this pipeline using **GitHub Actions**. 
GitHub Actions provides incredibly powerful build runners (2-core CPU, 7GB RAM) completely for free, which fits our needs perfectly. 

## Proposed Changes

We will create an automated GitHub Actions deployment. Instead of paying for a server that is idle 99% of the time, the workflow runs on a schedule (or manual trigger), spins up your exact Airflow environment, triggers the DAG, saves the model artifacts, and tears it down gracefully. 

---

### GitHub Actions Workflow

#### [NEW] .github/workflows/deploy_pipeline.yml
Create an automation pipeline that runs the following script:
1. **Boot environment**: Runs `docker-compose up -d`
2. **Health checks**: Waits for Airflow Webserver and Postgres to become fully healthy using a small bash poller.
3. **Trigger DAG**: Uses `docker exec` to trigger `logistics_pipeline_dag` using the Airflow CLI.
4. **Monitor execution**: Continuously polls the Airflow CLI until the DAG run reaches `success` or `failed` state.
5. **Persist artifacts**: Automatically uploads the generated MLflow tracking logs and the resulting model (`data/models/`) as a GitHub Artifact, giving you a full audit trail entirely for free. The model artifacts will be downloadable straight from your GitHub repo.

## Open Questions

> [!IMPORTANT]
> 1. To use this solution, you will need to push this repository `D:\portfolio` to GitHub. The GitHub Actions workflow will automatically enable itself once pushed to your repo. Are you okay with pushing this to GitHub?
> 2. Would you like the GitHub Action to run automatically on a regular schedule (e.g., every day at midnight) or should it only run when you manually trigger it? (I will configure it to be manually triggered by default using `workflow_dispatch`).

## Verification Plan

### Automated Tests
I will use PowerShell within our scratch environment to simulate the steps of the GitHub Action:
1. Run `docker-compose up -d`
2. Run Airflow CLI command to trigger the run
3. Verify that the outputs in `data/models` are correctly populated.
4. Tear down `docker-compose down`.

### Manual Verification
Once we test it locally and ensure the Docker sequence runs correctly via command-line (which mimics the GH actions environment), I will provide you with instructions on how to initialize the Git repository and push this to your GitHub account to see the zero-cost pipeline run perfectly!
