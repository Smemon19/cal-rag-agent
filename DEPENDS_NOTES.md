Dependencies are pinned in `requirements.txt` and are the single source of truth.
All installs (local and CI/CD/Cloud Run) must use this file.
Project runtime is standardized on Python 3.11.
If tooling needs a version file, set it to Python 3.11.x.
Any dependency change must be updated in `requirements.txt`.
Test changes in a fresh virtual environment before deployment.
The embedding model is baked into the Docker image at build time, eliminating Hugging Face cache errors on Cloud Run cold starts.
