# Deep Learning Mini-Project Plan

## 1. Project Constraints & Guidelines
Based on the `project.txt` and `course-description.txt` requirements, the selected project must strictly adhere to:
*   **Allowed Architectures:** LLMs (multimodal), Diffusion Models, GANs, Transformers (Vision/Audio), or recent Autoencoders.
*   **Data Restrictions:** **NO TEXT** as the primary data modality. Text can only be used as auxiliary data (e.g., text-to-image prompts). Acceptable modalities include Image, Video, Audio/Speech, Time-Series (ECG, sensors), etc.
*   **Deliverables:**
    1.  A functional application (web-based or desktop) to perform predictions on new user instances.
    2.  A detailed report (architecture, dataset, code, explanations, discussion, conclusion).
    3.  A GitHub repository containing all code.
*   **Course Alignment:** The project should preferably leverage knowledge from optimizing DNNs, CNNs, RNNs, Transfer Learning, and Dimensionality Reduction.

## 2. Where to Find Innovative Deep Learning Projects
To find a project that is innovative yet feasible within an academic semester, the following platforms are highly recommended:
*   **PapersWithCode (paperswithcode.com):** Filter by "Computer Vision", "Audio", or "Medical" tasks. Look for tasks with available datasets and official implementation code to ensure feasibility.
*   **Hugging Face Spaces & Models (huggingface.co):** Browse the trending "Vision", "Audio", or "Multimodal" spaces. You can find inspiration for web applications and pre-trained models for Transfer Learning.
*   **Kaggle Competitions (kaggle.com):** Look for completed competitions in medical imaging (e.g., anomaly detection) or audio processing. Kaggle provides ready-to-use datasets and community notebooks for baselines.
*   **GitHub Trending (github.com/trending):** Filter by Jupyter Notebook or Python. Look for tags like `diffusion-models`, `vision-transformers`, or `gans`.

## 3. Recommended Project Themes (Aligning with Course & Constraints)
*   **Theme A: Generative Audio/Music Synthesis (Transformers/Diffusion)**
    *   *Concept:* Generate short musical clips or sound effects given a category or minimal auxiliary text prompt.
    *   *Why:* Uses Transformers/Diffusion, avoids main text processing, highly innovative.
*   **Theme B: Medical Image Anomaly Detection (Autoencoders/GANs)**
    *   *Concept:* Detect anomalies in X-rays, MRIs, or ECG signals using recent autoencoders or GAN-based reconstruction.
    *   *Why:* Directly mentioned in the prompt ("ECG signal processing", "anomaly detection"), practical, uses dimensionality reduction concepts.
*   **Theme C: Advanced Style Transfer or Inpainting (Diffusion/GANs)**
    *   *Concept:* A web app where users upload an image and apply complex styles or remove objects smoothly.
    *   *Why:* Visual, impressive web app potential, leverages CNNs and GAN/Diffusion.

## 4. Initial Project Workflow
1.  **Phase 1: Discovery & Selection (Current)** - Scrape platforms for trending, viable project ideas. Select one that fits constraints.
2.  **Phase 2: Data Acquisition & Baseline** - Download the relevant dataset. Build a simple baseline model (e.g., standard CNN) to ensure the data pipeline works.
3.  **Phase 3: Advanced Architecture Implementation** - Implement the core advanced architecture (Transformer, Diffusion, or GAN). Apply Transfer Learning if training from scratch is too computationally expensive.
4.  **Phase 4: Optimization & Evaluation** - Tune hyperparameters, apply dimensionality reduction if needed, and evaluate results for the report.
5.  **Phase 5: Application Deployment** - Wrap the trained model in a web interface (Streamlit, Gradio, or FastAPI backend with a basic frontend).
6.  **Phase 6: Documentation & Submission** - Write the final report, clean up the Jupyter notebooks, and push to GitHub.

---

## 5. Web Scraping Prompt for the Next Agent
*Provide the following prompt to the web-scraping agent to gather concrete project ideas:*

> **Role:** Deep Learning Research Assistant
>
> **Task:** Scrape and identify 3 to 5 highly innovative and feasible Deep Learning project ideas for a university mini-project. 
> 
> **Where to Search:**
> 1. Hugging Face Spaces / Trending Models (filter for Vision, Audio, Time-Series).
> 2. PapersWithCode (browse state-of-the-art tasks in Computer Vision, Medical, Audio).
> 3. Kaggle (recent image or audio competitions with public datasets).
>
> **Strict Constraints:**
> - **NO NLP/TEXT as the primary modality.** Text can only be used as a supplementary condition (e.g., text-to-audio). 
> - Architectures must involve at least one of the following: Diffusion Models, GANs, Vision/Audio Transformers, or recent Autoencoders.
> - Must be feasible for a group of students to complete in a few weeks (ideally utilizing Transfer Learning or existing pre-trained backbones).
> - Needs a clear path to being deployed as a web application (e.g., an app where a user uploads an image/audio/signal and gets a prediction or generated output).
>
> **What to Return for Each Project Idea:**
> 1. **Project Title:**
> 2. **Core Idea:** A brief description of what the model does.
> 3. **Architecture Used:** (e.g., Vision Transformer, Stable Diffusion, etc.)
> 4. **Suggested Dataset:** Link or name of a publicly available dataset.
> 5. **Web App Concept:** How this would look as a deployed application.
> 6. **Source Link:** Where you found the inspiration (e.g., Kaggle, HF, PapersWithCode).
