---
name: nexus-create-pipeline
description: Create a new nexus pipeline
---

Look at the nexus-pipeline-definitions skill to understand the pipeline definition yaml file, and the nexus-pipeline-actors skill to understand pipeline actors.


# Development Workflow

When developing a pipeline, follow this progression from local development to cloud deployment:

## 1. Local Development with Local Actor References

Start with local code references in your pipeline definition:

```yaml
actor:
  repo: ~/projects/ml-pipelines
  path: actors.my-actor/actors.module:function
```

## 2. Test with run-local-venv

First, test the pipeline using `run-local-venv` which runs actors in your local Python environment:

```bash
pipeline_manager run-local-venv --pipeline-path path/to/pipeline.yaml path/to/invocation.yaml
```

This is fastest for iteration and debugging.

## 3. Test with run-local

Next, test with `run-local` which builds and runs actors in Docker containers locally:

```bash
pipeline_manager run-local --pipeline-path path/to/pipeline.yaml path/to/invocation.yaml
```

This validates that the containerized environment works correctly.

## 4. Commit and Update to Git References

Once local testing succeeds:

1. Commit your actor code changes to the ml-pipelines repo
2. Push to remote and note the commit hash
3. Update your pipeline definition to use git repo-hash references:

```yaml
actor:
  repo: https://github.com/data-rock/ml-pipelines.git@abc12345
  path: actors.my-actor/actors.module:function
```

## 5. Deploy to Cloud

Docker builds use `--ssh=default` for private git repo access. Ensure `ssh-agent` is running before deploying:

```bash
eval $(ssh-agent -c) && ssh-add   # fish shell
eval $(ssh-agent -s) && ssh-add   # bash/zsh
```

Use one of these approaches to deploy:

**Option A: deploy-pipeline (simple)**
```bash
pipeline_manager deploy-pipeline --pipeline-path path/to/pipeline.yaml
```

**Option B: create-pipeline + register-pipeline (for newer compiler features)**
```bash
pipeline_manager create-pipeline --pipeline-path path/to/pipeline.yaml
pipeline_manager register-pipeline --pipeline-path path/to/pipeline.yaml
```

# Steps for Creating a New Pipeline

1. Create a folder for the pipeline in the `ml-pipelines` repo under `pipelines/<new-pipeline-name>/<pipeline-version>`. Start with version `v0.1.0`.

2. Decide what pipeline actors you will need in your pipeline graph. Look through the `ml-pipelines/actors` to find any existing actors that you may be able to reuse.

3. Create any new pipeline actors you will need using the nexus-pipeline-actors skill.

4. Create a pipeline definition using the nexus-pipeline-definitions skill at `pipeline.yaml`

5. Create some test invocations for the new pipeline under `tests/` in the pipeline's folder.

6. Follow the Development Workflow above to test and deploy.


## Notes

- All nexus pipelines are created in their own folder under `ml-pipelines/pipelines/`, and every version has its own folder under that, e.g. `ml-pipelines/pipelines/my_new_pipeline/v0.1.0`
- The pipeline definition file is named `pipeline.yaml`, e.g. `ml-pipelines/pipelines/my_new_pipeline/v0.1.0/pipeline.yaml`
