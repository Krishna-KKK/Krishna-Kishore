import os
import io
import math
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="AI Transformation Rule Generator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_result_df: Optional[pd.DataFrame] = None
last_rules: Optional[str] = None



def to_json_safe(obj: Any) -> Any:
    """Safely converts Pandas/Numpy objects into JSON-serializable data."""
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, pd.Series):
        return obj.replace([np.inf, -np.inf], np.nan).tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (dict, list, tuple)):
        return json.loads(json.dumps(obj, default=str))
    if pd.isna(obj):
        return None
    return obj


def load_file(file: UploadFile):
    """Loads CSV/XLSX/JSON file into a clean Pandas DataFrame."""
    try:
        filename = file.filename.lower()
        content = file.file.read()
        file.file.seek(0)

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), parse_dates=True)
        elif filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            raise ValueError("Unsupported file format")

        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        df = df.dropna(how="all", axis=1)
        return df

    except Exception as e:
        print("❌ load_file() error:", e)
        return None


def describe_dataframe_flat(df: pd.DataFrame, label: str = "dataset") -> List[Dict[str, Any]]:
    """Produces structured metadata summary for a DataFrame."""
    if df is None or df.empty:
        return [{
            "Column": "⚠️ No data", "Type": "-", "Nulls": "-", "% Null": "-",
            "Unique": "-", "Samples": "-", "Numeric Summary": "-"
        }]

    result = []
    total_rows = len(df)
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nulls = int(s.isnull().sum())
        pct_null = round((nulls / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_vals = int(s.nunique(dropna=True))
        samples = s.dropna().astype(str).unique()[:5].tolist()

        numeric_summary = "-"
        if np.issubdtype(s.dtype, np.number) and s.notna().any():
            numeric_summary = f"mean:{s.mean():.2f} std:{s.std():.2f} min:{s.min()} max:{s.max()}"

        result.append({
            "Column": col, "Type": dtype, "Nulls": nulls, "% Null": pct_null,
            "Unique": unique_vals, "Samples": samples, "Numeric Summary": numeric_summary
        })
    return result


def mapping_rules_to_markdown(mapping_obj):
    """Converts mapping list into Markdown table."""
    if isinstance(mapping_obj, str):
        return mapping_obj
    if isinstance(mapping_obj, list):
        lines = ["| Source Column | Target Column | Transformation |", "|------|--------|--------|"]
        for rule in mapping_obj:
            src = rule.get("source_column", "")
            tgt = rule.get("target_column", "")
            trans = rule.get("transformation", "")
            lines.append(f"| {src} | {tgt} | {trans} |")
        return "\n".join(lines)
    return str(mapping_obj)


def generate_python_snippet(mapping_rules: list) -> str:
    """Generates Python data transformation code snippet."""
    lines = ["import pandas as pd", "", "def transform_data(df):"]
    lines.append("    df_transformed = pd.DataFrame()")
    for rule in mapping_rules:
        src = rule.get("source_column", "")
        tgt = rule.get("target_column", "")
        trans = rule.get("transformation", "direct_copy")
        if "Concatenate" in trans:
            prefix = trans.split("'")[1] if "'" in trans else ""
            lines.append(f"    df_transformed['{tgt}'] = '{prefix}' + df['{src}'].astype(str)")
        elif "Convert" in trans:
            lines.append(f"    df_transformed['{tgt}'] = df['{src}'].astype(str)")
        else:
            lines.append(f"    df_transformed['{tgt}'] = df['{src}']")
    lines.append("    return df_transformed")
    return "\n".join(lines)


def generate_sql_snippet(mapping_rules: list, source_table="source_table") -> str:
    """Generates SQL SELECT statement for transformations."""
    select_clauses = []
    for rule in mapping_rules:
        src = rule.get("source_column", "")
        tgt = rule.get("target_column", "")
        trans = rule.get("transformation", "direct_copy")
        if "Concatenate" in trans:
            prefix = trans.split("'")[1] if "'" in trans else ""
            select_clauses.append(f"CONCAT('{prefix}', {src}) AS `{tgt}`")
        elif "Convert" in trans:
            select_clauses.append(f"CAST({src} AS CHAR) AS `{tgt}`")
        else:
            select_clauses.append(f"{src} AS `{tgt}`")
    return f"SELECT {', '.join(select_clauses)} FROM {source_table};"


def build_expected_results_preview(df_source: pd.DataFrame, df_target: pd.DataFrame, mapping_rules: Optional[list] = None) -> Dict[str, Any]:
    """Builds preview of expected transformed output."""
    if df_source is None or df_source.empty:
        return {"note": "No source data provided.", "preview": []}

    expected = pd.DataFrame(index=df_source.index)
    if mapping_rules:
        for rule in mapping_rules:
            src = rule.get("source_column")
            tgt = rule.get("target_column")
            trans = rule.get("transformation", "direct_copy")
            if src in df_source.columns:
                if "Concatenate" in trans:
                    prefix = trans.split("'")[1] if "'" in trans else ""
                    expected[tgt] = prefix + df_source[src].astype(str)
                elif "Convert" in trans:
                    expected[tgt] = df_source[src].astype(str)
                else:
                    expected[tgt] = df_source[src]
            else:
                expected[tgt] = None
    else:
        for c in df_target.columns:
            expected[c] = df_source[c] if c in df_source.columns else np.nan

    return {"note": "Projected to target schema.", "preview": to_json_safe(expected.head(5))}


def get_source_df(source_file, last_result_df):
    """Decides whether to load file or reuse last DataFrame."""
    if source_file:
        df = load_file(source_file)
        return df if df is not None else pd.DataFrame()
    elif last_result_df is not None and not last_result_df.empty:
        return last_result_df
    else:
        return pd.DataFrame()


def call_openai_chat_with_retry(messages, model="gpt-4o-mini", retries=2):
    """Retries OpenAI API calls with exponential backoff."""
    if client is None:
        return None
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=0)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise


@app.post("/analyze")
async def analyze(source_file: UploadFile = File(None), target_file: UploadFile = File(None), query: str = Form(""), mode: str = Form("query")):
    global last_result_df, last_rules
    try:
        df_source = get_source_df(source_file, last_result_df)
        df_target = load_file(target_file) if target_file else pd.DataFrame()
        source_summary = describe_dataframe_flat(df_source)
        target_summary = describe_dataframe_flat(df_target)

        mapping_rules_list = []

        # AUTO MODE → use OpenAI
        if mode == "auto" and client:
            context = (
                f"You are an expert Data Transformation and Mapping AI.\n"
                f"Analyze source and target datasets and generate realistic transformation mappings.\n"
                f"Include conversions, renamings, concatenations, or derived fields when needed.\n\n"
                f"Source Data Summary: {json.dumps(source_summary[:5], indent=2)}\n"
                f"Target Data Summary: {json.dumps(target_summary[:5], indent=2)}\n"
                f"User Query: {query}\n\n"
                f"Return valid JSON ONLY with key 'mapping_rules' as a list of objects with keys: source_column, target_column, transformation."
            )
            try:
                resp = call_openai_chat_with_retry(
                    [
                        {"role": "system", "content": "Return ONLY JSON with key 'mapping_rules'."},
                        {"role": "user", "content": context},
                    ],
                    model="gpt-4o-mini",
                )
                text = resp.choices[0].message.content.strip()
                structured_output = json.loads(text)
                mapping_rules_list = structured_output.get("mapping_rules", [])
            except Exception:
                mapping_rules_list = []
        else:
            # QUERY MODE → basic fallback mapping
            mapping_rules_list = [
                {"source_column": c, "target_column": c, "transformation": "direct_copy"}
                for c in df_source.columns
            ]

        # Intelligent fallback for empty or invalid rules
        if not mapping_rules_list:
            mapping_rules_list = []
            for col in df_source.columns:
                trans_type = "direct_copy"
                if "date" in col.lower():
                    trans_type = "Convert to string format YYYY-MM-DD"
                elif "id" in col.lower():
                    trans_type = "Ensure as integer ID"
                elif "price" in col.lower() or "amount" in col.lower():
                    trans_type = "Convert to float and round to 2 decimals"
                mapping_rules_list.append({
                    "source_column": col,
                    "target_column": col,
                    "transformation": trans_type
                })

        expected = build_expected_results_preview(df_source, df_target, mapping_rules_list)
        sql_query = generate_sql_snippet(mapping_rules_list)
        python_code = generate_python_snippet(mapping_rules_list)

        payload = {
            "source_analysis": source_summary,
            "target_analysis": target_summary,
            "mapping_rules": mapping_rules_to_markdown(mapping_rules_list),
            "sql_transformation_query": sql_query,
            "python_code_snippet": python_code,
            "expected_results": expected,
        }

        last_rules = sql_query
        last_result_df = df_source.copy()
        return JSONResponse(to_json_safe(payload))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/chat")
async def chat(message: Optional[str] = Form(None), source_file: UploadFile = File(None), target_file: UploadFile = File(None)):
    global last_result_df, last_rules
    try:
        user_msg = message or "Generate transformation mappings."
        df_source = get_source_df(source_file, last_result_df)
        df_target = load_file(target_file) if target_file else pd.DataFrame()

        source_summary = describe_dataframe_flat(df_source)
        target_summary = describe_dataframe_flat(df_target)

        context = (
            f"You are an expert Data Transformation and Mapping AI.\n"
            f"Analyze source and target datasets and generate realistic mapping rules.\n"
            f"Include renaming, datatype conversions, concatenations, and derived fields when appropriate.\n\n"
            f"--- SOURCE DATASET ---\n{json.dumps(source_summary[:5], indent=2)}\n\n"
            f"--- TARGET DATASET ---\n{json.dumps(target_summary[:5], indent=2)}\n\n"
            f"User Query: {user_msg}\n\n"
            f"Return valid JSON ONLY with key 'mapping_rules' as a list of objects, each having: source_column, target_column, transformation."
        )

        mapping_rules_list: List[dict] = []

        if client:
            try:
                resp = call_openai_chat_with_retry(
                    [
                        {"role": "system", "content": "Return ONLY JSON with key 'mapping_rules'."},
                        {"role": "user", "content": context},
                    ],
                    model="gpt-4o-mini",
                )
                text = resp.choices[0].message.content.strip()
                structured_output = json.loads(text)
                mapping_rules_list = structured_output.get("mapping_rules", [])
            except Exception as e:
                print("⚠️ OpenAI Chat Mode error:", e)
                mapping_rules_list = []
        else:
            mapping_rules_list = []

        # Intelligent fallback
        if not mapping_rules_list:
            mapping_rules_list = []
            for col in df_source.columns:
                trans_type = "direct_copy"
                if "date" in col.lower():
                    trans_type = "Convert to string format YYYY-MM-DD"
                elif "id" in col.lower():
                    trans_type = "Ensure as integer ID"
                elif "price" in col.lower() or "amount" in col.lower():
                    trans_type = "Convert to float and round to 2 decimals"
                mapping_rules_list.append({
                    "source_column": col,
                    "target_column": col,
                    "transformation": trans_type
                })

        expected = build_expected_results_preview(df_source, df_target, mapping_rules_list)
        sql_query = generate_sql_snippet(mapping_rules_list)
        python_code = generate_python_snippet(mapping_rules_list)

        payload = {
            "source_analysis": source_summary,
            "target_analysis": target_summary,
            "mapping_rules": mapping_rules_to_markdown(mapping_rules_list),
            "sql_transformation_query": sql_query,
            "python_code_snippet": python_code,
            "expected_results": expected,
        }

        last_rules = sql_query
        last_result_df = df_source.copy()
        return JSONResponse(to_json_safe(payload))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    
# ✅ Download Excel endpoint
@app.get("/download-excel")
async def download_excel():
    global last_result_df
    if last_result_df is None or last_result_df.empty:
        return JSONResponse({"error": "No transformed dataset available. Please analyze first."}, status_code=400)

    file_path = os.path.join(BASE_DIR, "transformed_output.xlsx")
    last_result_df.to_excel(file_path, index=False)

    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="transformed_output.xlsx"
    )


# ✅ Download SQL endpoint
@app.get("/download-sql")
async def download_sql():
    global last_rules
    if not last_rules:
        return JSONResponse({"error": "No SQL rules generated. Please analyze first."}, status_code=400)

    file_path = os.path.join(BASE_DIR, "transformation.sql")
    with open(file_path, "w") as f:
        f.write(last_rules)

    return FileResponse(
        file_path,
        media_type="application/sql",
        filename="transformation.sql"
    )


@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "API is running."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8500)
