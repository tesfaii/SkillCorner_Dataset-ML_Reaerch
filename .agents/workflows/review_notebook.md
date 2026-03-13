---
description: Review and standardize a new Jupyter Notebook
---
# Jupyter Notebook Review Workflow

This workflow ensures that any new Jupyter Notebook added to the repository conforms to our standard conventions for structure, readability, and variable naming.

1. **Read the Notebook from Staging**
   Look for the newly added `.ipynb` file in the `notebooks/exploratory/` directory. Use the `view_file` tool to inspect the contents of the `.ipynb` file. Keep in mind that notebooks are structured as JSON files.

2. **Check for Useless Cells**
   - Identify and carefully remove any completely empty cells (code or markdown).
   - Identify and remove cells that contain only commented-out code with no explanatory value, or arbitrary scratchpad code (e.g. `black .`).

3. **Audit Package Dependencies**
   - Review all `import` and `from ... import ...` statements.
   - Verify that each external package is publicly available and open-source (e.g., on PyPI).
   - If you encounter a local or company-specific proprietary package (e.g., `skillcorner.client`), use the `notify_user` tool to ask the user what the package or function does. Your goal is to replace the proprietary import with a local python function that replicates the behavior so that external users can run the notebook without internal credentials.

4. **Scrub Credentials and PII**
   - Review code cells for any hardcoded credentials, API keys, user passwords, or Personally Identifiable Information (PII).
   - If found, redact the sensitive data, replacing it with placeholder values like `"YOUR_API_KEY"` or environment variable calls (e.g., `os.environ.get('API_KEY')`).

5. **Verify Markdown Coverage and Structure**
   - Ensure the notebook uses a single level 1 heading for the title and level 2 headings for major steps (e.g., `## 📋 Step 1: Setup & Prerequisites`, `## 📥 Step 2: Load Data`).
   - Verify that all complex code blocks are accompanied by a descriptive markdown cell preceding them. Add comprehensive markdown text if cells are under-documented.

6. **Standardize Variable Naming**
   - Verify that the notebook uses standard data pipeline variable names.
   - Adopt standard base names for common dataframes, such as `raw_df`, `tracking_df`, `players_df`, `events_df`, `enriched_df`, `filtered_df`, and `aggregated_df`.
   - Use `match_id` consistently for match identifiers.

7. **Apply Fixes Carefully**
   - Use the `replace_file_content` or `multi_replace_file_content` tools to make the necessary text edits to the notebook's JSON structure or source code.
   - **Important:** Ensure you do not break the valid JSON format of the `.ipynb` file while making edits. Always verify brackets and commas.

8. **Final Validation and Relocation**
   - Review your changes to confirm the notebook is clean, well-documented, follows naming conventions, and doesn't contain leftover blank spaces.
   - Once standardized, advise the user that the notebook is ready to be moved from `notebooks/exploratory/` to its final directory (e.g., `notebooks/tutorials/`). If you are unsure whether the notebook is a tutorial or belongs elsewhere, explicitly ask the orchestrator where it should be moved.

9. **Generate a Flow Diagram**
   To help visualize the notebook's logic, use the following prompt to ask another agent (or an LLM) to generate a flowchart:
   
   *Prompt for diagram generation:*
   > "Please analyze the provided Jupyter Notebook and generate a Mermaid flowchart diagram that explains the flow of data and operations. The diagram should capture the major logical sections as identified by the numbered markdown headings (e.g., Data Extraction, Metadata Loading, Merging). Use the established variable names (e.g., `raw_data`, `tracking_df`, `players_df`, `enriched_df`) to label the transitions and output nodes. Ensure the diagram is concise but comprehensively summarizes the notebook's end-to-end data pipeline."
