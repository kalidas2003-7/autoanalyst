"""
chat_app.py — "Chat interface" stage.

Streamlit front-end: upload multiple files, chat in natural language,
AutoAnalystAgent runs the full plan -> tools -> EDA/ML/viz -> report
pipeline each turn, with persistent memory across the session.

Run with:  streamlit run chat_app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from autoanalyst.orchestrator import AutoAnalystAgent

st.set_page_config(page_title="AutoAnalyst", page_icon="📊", layout="wide")

st.title("📊 AutoAnalyst — LLM/Agent-as-Data-Analyst")
st.caption("Multiple files → Agent planner → Tool selection → Python execution → "
           "EDA → ML → Visualization → Report generation → Chat → Memory")

if "agent" not in st.session_state:
    st.session_state.agent = AutoAnalystAgent(workdir=os.path.dirname(__file__) or ".")
    st.session_state.messages = []

agent: AutoAnalystAgent = st.session_state.agent

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader(
        "CSV, XLSX, JSON, Parquet, TXT — multiple files supported",
        accept_multiple_files=True,
        type=["csv", "xlsx", "xls", "json", "jsonl", "parquet", "txt", "tsv"],
    )
    if uploaded and st.button("Load files"):
        upload_dir = os.path.join(os.path.dirname(__file__) or ".", "uploads", agent.session_id)
        os.makedirs(upload_dir, exist_ok=True)
        paths = []
        for uf in uploaded:
            p = os.path.join(upload_dir, uf.name)
            with open(p, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(p)
        manifest = agent.load_files(paths)
        st.success(f"Loaded {len(paths)} file(s).")
        st.json(manifest)

    st.divider()
    st.header("Session")
    st.code(agent.session_id)
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("LLM planner active (Claude)")
    else:
        st.info("Rule-based planner active (set ANTHROPIC_API_KEY to enable LLM planning)")
    if st.button("Reset conversation"):
        agent.memory.clear()
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data, e.g. 'explore this dataset' or "
                            "'predict churn from the customers table'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not agent.loader.files:
            reply = "Please upload at least one data file in the sidebar first."
            st.markdown(reply)
        else:
            with st.spinner("Planning and running the analysis pipeline..."):
                result = agent.chat(prompt)

            with st.expander("🧭 Plan"):
                st.json(result.plan)

            st.markdown(result.report_markdown or "No report generated.")

            for sr in result.step_results:
                out = sr["output"]
                if isinstance(out, dict) and out.get("figure_path"):
                    st.image(out["figure_path"])
                if isinstance(out, dict) and out.get("paths"):
                    for p in out["paths"]:
                        if os.path.exists(p):
                            st.image(p)

            if result.errors:
                with st.expander("⚠️ Errors during execution"):
                    st.json(result.errors)

            reply = result.report_markdown or "Done."

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.divider()
with st.expander("🐍 Ad-hoc Python execution"):
    code = st.text_area("Run pandas/sklearn code against `df` (the loaded data):",
                         "result = df.describe()")
    if st.button("Run code"):
        if not agent.loader.files:
            st.warning("Upload data first.")
        else:
            out = agent.run_code(code)
            if out["success"]:
                st.text(out["stdout"])
                if out["result"] is not None:
                    st.write(out["result"])
                if out["figure_path"]:
                    st.image(out["figure_path"])
            else:
                st.error(out["error"])
